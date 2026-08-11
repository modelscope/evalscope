"""Resolve canonical metric identities into persisted display semantics.

V2 lookups use ``MetricIdentity.name`` plus an optional ``(name, aggregation)`` override. Dynamic
axes such as ``k``, scope, threshold and category remain structured dimensions and therefore do
not expand the registry. Historical semantic anchors are consulted only while migrating old
reports. Unknown third-party metrics degrade to diagnostics without changing their values.

Primary role assignment is intentionally separate and happens once per report through
``attribute_metric_roles``. The resolver never reads a data adapter and never selects a primary.
"""

import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

from evalscope.api.metric.semantics import (
    MetricDirection,
    MetricDisplayKind,
    MetricEntry,
    MetricIdentity,
    MetricRole,
    MetricSelector,
    MetricSemantics,
)
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
from evalscope.metrics.semantics.catalog import (
    AGGREGATION_SEMANTICS,
    BENCHMARK_METRIC_OVERRIDES,
    METRIC_DEFINITIONS,
    METRIC_NAME_TABLE_LOCATION,
)
from evalscope.metrics.semantics.formatting import DIAGNOSTIC_FALLBACK_PRECISION
from evalscope.metrics.semantics.identity import migrate_legacy_identity
from evalscope.utils import get_logger

if TYPE_CHECKING:
    from evalscope.report.report import Report

logger = get_logger()

#: Prefix of every audit message emitted by this module, greppable in logs.
AUDIT_MESSAGE_PREFIX = '[metric-semantics]'

#: ``semantic_id`` used when no source of the priority chain matched.
DIAGNOSTIC_FALLBACK_SEMANTIC_ID = 'diagnostic.unspecified'

#: Where perf field semantics are declared, used in audit messages.
PERF_FIELD_TABLE_LOCATION = 'evalscope/metrics/semantics/perf.py::PERF_SEMANTICS'

#: Directory holding one JSON file per built-in benchmark. Resolved without importing
#: ``evalscope.utils.resource_utils`` to keep this module cheap to import.
_BUILTIN_META_DIR = Path(__file__).parents[2] / 'benchmarks' / '_meta'

__all__ = [
    'AUDIT_MESSAGE_PREFIX',
    'DIAGNOSTIC_FALLBACK_SEMANTIC_ID',
    'ResolvedSemantics',
    'SemanticsResolver',
    'SemanticsSource',
    'attribute_metric_roles',
    'catalog_entry_location',
    'diagnostic_fallback',
    'get_semantics_resolver',
    'hydrate_report_semantics',
]


class SemanticsSource(str, Enum):
    """Which level of the fixed priority chain produced a resolution."""

    REPORT_ANCHOR = 'report_anchor'
    """A ``semantic_id`` anchor stored in the report, materialized from the baseline table."""

    BENCHMARK_OVERRIDE = 'benchmark_override'
    """A ``(benchmark, metric)`` collision override in the catalog."""

    METRIC_NAME = 'metric_name'
    """The final report metric name is declared in the metric name table."""

    DIAGNOSTIC_FALLBACK = 'diagnostic_fallback'
    """Nothing matched: ``diagnostic.unspecified`` with the raw value kept as is."""


class ResolvedSemantics(BaseModel):
    """Outcome of one resolution: the semantics, its source and the audit trail."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    semantics: MetricSemantics
    """The resolved contract. Always present, even for a degradation."""

    source: SemanticsSource
    """Level of the priority chain the semantics came from."""

    audit_messages: List[str] = Field(default_factory=list)
    """Human readable messages naming the metric and where to declare it."""

    @property
    def degraded(self) -> bool:
        """Whether the diagnostic fallback was used instead of a declared semantics."""
        return self.source is SemanticsSource.DIAGNOSTIC_FALLBACK

    def log_audit_messages(self) -> None:
        """Log the audit trail as warnings, naming each metric and where to declare it."""
        for message in self.audit_messages:
            logger.warning(message)


def diagnostic_fallback(metric_name: str) -> MetricSemantics:
    """Build the diagnostic semantics used when no source of the chain matched.

    The value is shown exactly as stored: no direction, no unit, no display multiplier, no
    range, so no consumer can present it as good or bad.

    Args:
        metric_name: Final report metric name, or perf field key, that failed to resolve.

    Returns:
        Semantics with ``semantic_id='diagnostic.unspecified'`` and ``role=diagnostic``.
    """
    return MetricSemantics(
        semantic_id=DIAGNOSTIC_FALLBACK_SEMANTIC_ID,
        metric_name=metric_name,
        role=MetricRole.DIAGNOSTIC,
        direction=MetricDirection.NONE,
        display_kind=MetricDisplayKind.NUMBER,
        display_multiplier=None,
        display_unit=None,
        display_precision=DIAGNOSTIC_FALLBACK_PRECISION,
    )


def catalog_entry_location(final_metric_name: str) -> str:
    """Describe where to declare a canonical metric name in the registry.

    Args:
        final_metric_name: Canonical metric name that failed to resolve.

    Returns:
        A path of the form ``...catalog.py::METRIC_DEFINITIONS['accuracy']``.
    """
    return f"{METRIC_NAME_TABLE_LOCATION}['{final_metric_name}']"


def _undeclared_metric_message(benchmark_name: str, identity: MetricIdentity) -> str:
    """Format the audit message of a metric that resolved to the diagnostic fallback."""
    return (
        f"{AUDIT_MESSAGE_PREFIX} undeclared metric: benchmark='{benchmark_name}' "
        f"identity='{identity.key}'\n  add an entry at {catalog_entry_location(identity.name)}"
    )


def _undeclared_perf_field_message(field_key: str) -> str:
    """Format the audit message of a perf field that resolved to the diagnostic fallback."""
    return (
        f"{AUDIT_MESSAGE_PREFIX} undeclared perf field: field_key='{field_key}'\n"
        f'  add an entry at {PERF_FIELD_TABLE_LOCATION}'
    )


@lru_cache(maxsize=None)
def _meta_primary_metric(benchmark_name: str) -> Optional[MetricSelector]:
    """Return the primary selector recorded in a benchmark's ``_meta`` file.

    Used to recover the primary metric of a legacy report. Reads only bundled metadata and never
    imports the adapter.

    Args:
        benchmark_name: Built-in benchmark name.

    Returns:
        The raw metric name declared as primary and the aggregation name, or ``None`` when
        unavailable.
    """
    path = _BUILTIN_META_DIR / f'{benchmark_name}.json'
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    meta = data.get('meta') if isinstance(data, dict) else None
    if not isinstance(meta, dict):
        return None
    primary = meta.get('primary_metric')
    if isinstance(primary, dict):
        try:
            return MetricSelector.model_validate(primary)
        except ValueError:
            return None
    if not isinstance(primary, str) or not primary:
        return None
    aggregation = meta.get('aggregation')
    identity = migrate_legacy_identity(
        primary, aggregation if isinstance(aggregation, str) else '', benchmark_name=benchmark_name
    )
    return MetricSelector(name=identity.name)


def _with_role(semantics: MetricSemantics, role: MetricRole) -> MetricSemantics:
    if semantics.role is role:
        return semantics
    return MetricSemantics(**{**semantics.model_dump(), 'role': role})


def attribute_metric_roles(
    identities: Sequence[MetricIdentity],
    semantics_by_identity: Mapping[str, MetricSemantics],
    selector: Optional[MetricSelector],
) -> Tuple[Dict[str, MetricSemantics], Optional[MetricIdentity]]:
    """Select exactly one primary identity and assign report roles once.

    An explicit selector must match exactly one emitted identity. Without a selector, implicit
    primary selection is allowed only when exactly one non-diagnostic identity exists.
    """
    if selector is not None:
        matches = [identity for identity in identities if selector.matches(identity)]
        if len(matches) != 1:
            raise ValueError(
                f'primary metric selector {selector.model_dump()} matched {len(matches)} identities; expected exactly one'
            )
        primary = matches[0]
        if semantics_by_identity[primary.key].role is MetricRole.DIAGNOSTIC:
            raise ValueError(f'primary metric selector matched diagnostic identity {primary.key}')
    else:
        graded = [
            identity for identity in identities if semantics_by_identity[identity.key].role is not MetricRole.DIAGNOSTIC
        ]
        if not graded:
            return dict(semantics_by_identity), None
        if len(graded) != 1:
            raise ValueError(
                f'benchmark emitted {len(graded)} non-diagnostic metric identities; declare BenchmarkMeta.primary_metric'
            )
        primary = graded[0]

    attributed: Dict[str, MetricSemantics] = {}
    for identity in identities:
        semantics = semantics_by_identity[identity.key]
        if semantics.role is MetricRole.DIAGNOSTIC:
            attributed[identity.key] = semantics
        else:
            role = MetricRole.PRIMARY if identity == primary else MetricRole.AUXILIARY
            attributed[identity.key] = _with_role(semantics, role)
    return attributed, primary


class SemanticsResolver:
    """Resolve canonical metric identities into base ``MetricSemantics``.

    The resolver is stateless apart from the tables it reads, so a single instance can be shared;
    use :func:`get_semantics_resolver` for the process-wide one. Tables are injectable to keep
    the resolution logic testable without touching the shipped catalog.
    """

    def __init__(
        self,
        name_table: Optional[Mapping[str, MetricEntry]] = None,
        aggregation_table: Optional[Mapping[Tuple[str, str], MetricEntry]] = None,
        override_table: Optional[Mapping[Tuple[str, str], MetricEntry]] = None,
        perf_fields: Optional[Mapping[str, MetricEntry]] = None,
    ) -> None:
        """Build a resolver.

        Args:
            name_table: Canonical metric name -> entry. Defaults to ``METRIC_DEFINITIONS``.
            aggregation_table: ``(metric, aggregation)`` semantic overrides.
            override_table: ``(benchmark, metric)`` -> entry. Defaults to the collision table.
            perf_fields: Perf field key -> entry. Defaults to the perf table, imported lazily so
                this module stays importable before that table exists.
        """
        self._names = METRIC_DEFINITIONS if name_table is None else name_table
        self._aggregations = AGGREGATION_SEMANTICS if aggregation_table is None else aggregation_table
        self._overrides = BENCHMARK_METRIC_OVERRIDES if override_table is None else override_table
        self._perf_fields = perf_fields

    def resolve(
        self,
        benchmark_name: str,
        identity: MetricIdentity,
        embedded_semantic_id: Optional[str] = None,
    ) -> ResolvedSemantics:
        """Resolve one identity without assigning its report-level role.

        Args:
            benchmark_name: Benchmark (dataset) the metric belongs to.
            identity: Canonical identity emitted by an aggregator.
            embedded_semantic_id: ``semantic_id`` anchor stored in the report. It is used when the
                current catalog has no declaration for ``final_metric_name``.
        Returns:
            The resolution, never ``None`` and never raising. An undeclared name degrades to the
            diagnostic fallback and carries the audit messages naming where to declare it.
        """
        metric_name = identity.name

        # 1. Benchmark level collision override.
        entry = self._overrides.get((benchmark_name, metric_name))
        if entry is not None:
            semantics = entry.resolve(metric_name)
            return ResolvedSemantics(semantics=semantics, source=SemanticsSource.BENCHMARK_OVERRIDE)

        # 2. Aggregation-specific override, then the canonical name table.
        entry = self._aggregations.get((metric_name, identity.aggregation))
        if entry is None:
            entry = self._names.get(metric_name)
        if entry is not None:
            semantics = entry.resolve(metric_name)
            if semantics.role is MetricRole.PRIMARY:
                semantics = _with_role(semantics, MetricRole.AUXILIARY)
            return ResolvedSemantics(semantics=semantics, source=SemanticsSource.METRIC_NAME)

        # 3. Report anchor: retain a historical declaration whose name is no longer catalogued.
        if embedded_semantic_id is not None:
            baseline = SEMANTIC_BASELINES.get(embedded_semantic_id)
            if baseline is not None:
                semantics = baseline
                if semantics.role is MetricRole.PRIMARY:
                    semantics = _with_role(semantics, MetricRole.AUXILIARY)
                return ResolvedSemantics(semantics=semantics, source=SemanticsSource.REPORT_ANCHOR)

        # 4. Diagnostic fallback: the value is shown as stored and the gap is logged.
        return ResolvedSemantics(
            semantics=diagnostic_fallback(metric_name),
            source=SemanticsSource.DIAGNOSTIC_FALLBACK,
            audit_messages=[_undeclared_metric_message(benchmark_name, identity)],
        )

    def resolve_perf_field(self, field_key: str) -> ResolvedSemantics:
        """Resolve one perf field key.

        Args:
            field_key: Key of the perf field, taken from the ``Metrics`` / ``PercentileMetrics``
                constants for the perf archive API, or the stable API path of an in-report perf
                metric.

        Returns:
            The resolution, never ``None`` and never raising.
        """
        entry = self._perf_field_entries().get(field_key)
        if entry is not None:
            return ResolvedSemantics(semantics=entry.resolve(field_key), source=SemanticsSource.METRIC_NAME)

        return ResolvedSemantics(
            semantics=diagnostic_fallback(field_key),
            source=SemanticsSource.DIAGNOSTIC_FALLBACK,
            audit_messages=[_undeclared_perf_field_message(field_key)],
        )

    def _perf_field_entries(self) -> Mapping[str, MetricEntry]:
        """Return the perf tables, imported lazily and tolerating their absence.

        Perf data reaches the API under three key spaces, all merged here: the display names of
        the perf constants (percentile and summary JSON), the stable API paths (in-report perf and
        the run list), and the archive summary table's column labels.
        """
        if self._perf_fields is not None:
            return self._perf_fields
        try:
            from evalscope.metrics.semantics.perf import PERF_SEMANTICS
        except ImportError:
            return {}
        return PERF_SEMANTICS


@lru_cache(maxsize=1)
def get_semantics_resolver() -> SemanticsResolver:
    """Return the process-wide resolver reading the shipped tables."""
    return SemanticsResolver()


def hydrate_report_semantics(report: 'Report') -> 'Report':
    """Migrate a historical report to persisted v2 semantics.

    Args:
        report: Report to hydrate, mutated in place.

    Returns:
        The same report instance.
    """
    metrics = list(getattr(report, 'metrics', None) or [])
    if not metrics:
        return report

    benchmark_name = getattr(report, 'dataset_name', '') or ''
    resolver = get_semantics_resolver()
    selector = _meta_primary_metric(benchmark_name)
    persisted_primary = getattr(report, 'primary_metric_identity', None)
    if persisted_primary is not None:
        selector = MetricSelector(
            name=persisted_primary.name,
            aggregation=persisted_primary.aggregation,
            dimensions=persisted_primary.dimensions,
        )

    identities = [metric.identity for metric in metrics]
    semantics_by_identity: Dict[str, MetricSemantics] = {}
    for metric in metrics:
        embedded_semantic_id = metric.semantics.semantic_id if metric.semantics else None
        resolved = resolver.resolve(benchmark_name, metric.identity, embedded_semantic_id=embedded_semantic_id)
        resolved.log_audit_messages()
        semantics_by_identity[metric.identity.key] = resolved.semantics
        if not resolved.degraded and resolved.semantics.role is not MetricRole.DIAGNOSTIC:
            metric.legacy_name = None

    try:
        semantics_by_identity, primary_identity = attribute_metric_roles(identities, semantics_by_identity, selector)
    except ValueError as error:
        logger.warning(f'{AUDIT_MESSAGE_PREFIX} legacy report has no unambiguous primary metric: {error}')
        primary_identity = None
    for metric in metrics:
        metric.semantics = semantics_by_identity[metric.identity.key]
    report.primary_metric_identity = primary_identity

    return report
