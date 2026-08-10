"""Metric semantics resolution.

``SemanticsResolver`` turns a *final report metric name* (the string ``ReportGenerator`` writes
into ``Metric.name``) into a ``MetricSemantics`` using one fixed priority chain, so a freshly
generated report, a historical report and the service APIs all agree on the direction, unit and
display rules of the same metric.

Priority chain, first hit wins:

1. ``BENCHMARK_OVERRIDE`` -- ``(benchmark_name, final_metric_name)`` has a collision override.
2. ``METRIC_NAME`` -- the final report metric name is declared in :data:`METRIC_NAME_SEMANTICS`.
3. ``REPORT_ANCHOR`` -- the report stores a ``semantic_id`` anchor and the current catalog has no
   declaration for its name. The baseline keeps renamed or removed catalog entries readable.
4. ``DIAGNOSTIC_FALLBACK`` -- nothing matched: ``diagnostic.unspecified``, the raw value is kept
   as is and an audit message records where to add the missing entry.

After any hit the resolver applies the benchmark level role adjustment: the benchmark's
``primary_metric`` (the final name, supplied by the generator, ``Report.primary_metric_name`` or
``_meta``) is promoted to ``primary`` and every other non-diagnostic metric is demoted to
``auxiliary``. This adjusts only the ``role`` field, it never introduces a new lookup level.

Every lookup is an exact dictionary lookup: no regular expressions, no name normalization, no
fuzzy matching and no inference from the magnitude or the range of a value.

Degrading, never blocking
-------------------------
Resolution never raises and never stops a caller: an undeclared name degrades to
``diagnostic.unspecified``, which shows the stored value without claiming a direction or a unit,
and logs where to declare it. Failing instead was tried and is not viable: a final metric name
embeds ``AggScore.aggregation_name``, which several benchmarks derive from the data (a subset
label in ``hallusion_bench``, a question type in ``longmemeval``, a needle range in
``openai_mrcr``), so the set of names a benchmark can emit is not knowable ahead of time and no
exact-key catalog can be complete. Degrading keeps those benchmarks reportable while the audit
log still names every gap worth closing.
"""

import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from evalscope.api.metric.semantics import MetricDirection, MetricDisplayKind, MetricEntry, MetricRole, MetricSemantics
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
from evalscope.metrics.semantics.catalog import (
    BENCHMARK_METRIC_OVERRIDES,
    METRIC_NAME_SEMANTICS,
    METRIC_NAME_TABLE_LOCATION,
)
from evalscope.metrics.semantics.formatting import DIAGNOSTIC_FALLBACK_PRECISION
from evalscope.metrics.semantics.naming import match_primary_final_name
from evalscope.utils import get_logger

if TYPE_CHECKING:
    from evalscope.report.report import Report

logger = get_logger()

#: Prefix of every audit message emitted by this module, greppable in logs.
AUDIT_MESSAGE_PREFIX = '[metric-semantics]'

#: ``semantic_id`` used when no source of the priority chain matched.
DIAGNOSTIC_FALLBACK_SEMANTIC_ID = 'diagnostic.unspecified'

#: Where perf field semantics are declared, used in audit messages.
PERF_FIELD_TABLE_LOCATION = 'evalscope/metrics/semantics/perf.py::PERF_FIELD_SEMANTICS'

#: Directory holding one JSON file per built-in benchmark. Resolved without importing
#: ``evalscope.utils.resource_utils`` to keep this module cheap to import.
_BUILTIN_META_DIR = Path(__file__).parents[2] / 'benchmarks' / '_meta'

__all__ = [
    'AUDIT_MESSAGE_PREFIX',
    'DIAGNOSTIC_FALLBACK_SEMANTIC_ID',
    'ResolvedSemantics',
    'SemanticsResolver',
    'SemanticsSource',
    'apply_primary_metric_roles',
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
    """Describe where to declare a final report metric name in the catalog.

    Args:
        final_metric_name: Metric name that failed to resolve.

    Returns:
        A path of the form ``...catalog.py::METRIC_NAME_SEMANTICS['mean_acc']``.
    """
    return f"{METRIC_NAME_TABLE_LOCATION}['{final_metric_name}']"


def _undeclared_metric_message(benchmark_name: str, final_metric_name: str) -> str:
    """Format the audit message of a metric that resolved to the diagnostic fallback."""
    return (
        f"{AUDIT_MESSAGE_PREFIX} undeclared metric: benchmark='{benchmark_name}' "
        f"metric='{final_metric_name}'\n  add an entry at {catalog_entry_location(final_metric_name)}"
    )


def _undeclared_perf_field_message(field_key: str) -> str:
    """Format the audit message of a perf field that resolved to the diagnostic fallback."""
    return (
        f"{AUDIT_MESSAGE_PREFIX} undeclared perf field: field_key='{field_key}'\n"
        f'  add an entry at {PERF_FIELD_TABLE_LOCATION}'
    )


@lru_cache(maxsize=None)
def _meta_primary_metric(benchmark_name: str) -> Optional[Tuple[str, str]]:
    """Return the ``(primary_metric, aggregation)`` recorded in a benchmark's ``_meta`` file.

    Used to recover the primary metric of a legacy report that predates
    ``Report.primary_metric_name``. Reads only the bundled metadata, never imports the adapter.

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
    if not isinstance(primary, str) or not primary:
        return None
    aggregation = meta.get('aggregation')
    return primary, aggregation if isinstance(aggregation, str) else ''


def _with_primary_role(
    semantics: MetricSemantics,
    final_metric_name: str,
    primary_metric_name: Optional[str],
) -> MetricSemantics:
    """Apply the benchmark level role adjustment to an already resolved semantics.

    When the benchmark declares a primary metric, the matching final name is promoted to
    ``primary`` and every other non-diagnostic metric is demoted to ``auxiliary``. Only the
    ``role`` field changes; a diagnostic metric is never promoted (that would contradict
    ``direction=none``), so an invalid promotion is left untouched.

    Args:
        semantics: Semantics resolved from the priority chain.
        final_metric_name: Final report metric name being resolved.
        primary_metric_name: The benchmark's primary metric as a final report name, or ``None``.

    Returns:
        The semantics with an adjusted ``role``, or the input unchanged.
    """
    if primary_metric_name is None:
        return semantics

    if final_metric_name == primary_metric_name:
        target_role = MetricRole.PRIMARY
    elif semantics.role is MetricRole.DIAGNOSTIC:
        return semantics
    else:
        target_role = MetricRole.AUXILIARY

    if semantics.role is target_role:
        return semantics
    try:
        return MetricSemantics(**{**semantics.model_dump(), 'role': target_role})
    except ValueError:
        # A diagnostic metric named as primary cannot carry role=primary (direction=none):
        # keep it as is rather than crash a read path.
        return semantics


def apply_primary_metric_roles(
    semantics_by_metric: Mapping[str, MetricSemantics],
    primary_metric_name: Optional[str],
) -> Dict[str, MetricSemantics]:
    """Attribute report-level roles after every emitted metric name is known.

    A single metric is implicitly primary. For a multi-metric report, an explicit declaration
    promotes exactly one graded metric and demotes the rest; without one, every graded metric is
    auxiliary so the report can choose a headline while marking that choice as inferred.

    Args:
        semantics_by_metric: Final report metric name -> resolved semantics.
        primary_metric_name: Explicit final primary name, or ``None``.

    Returns:
        A new mapping with report-level roles applied.
    """
    if primary_metric_name is None and len(semantics_by_metric) <= 1:
        return dict(semantics_by_metric)

    attributed: Dict[str, MetricSemantics] = {}
    for name, semantics in semantics_by_metric.items():
        if primary_metric_name is not None:
            attributed[name] = _with_primary_role(semantics, name, primary_metric_name)
            continue
        if semantics.role is MetricRole.DIAGNOSTIC or semantics.role is MetricRole.AUXILIARY:
            attributed[name] = semantics
            continue
        attributed[name] = MetricSemantics(**{**semantics.model_dump(), 'role': MetricRole.AUXILIARY})
    return attributed


class SemanticsResolver:
    """Resolve final report metric names into ``MetricSemantics`` with a fixed priority chain.

    The resolver is stateless apart from the tables it reads, so a single instance can be shared;
    use :func:`get_semantics_resolver` for the process-wide one. Tables are injectable to keep
    the resolution logic testable without touching the shipped catalog.
    """

    def __init__(
        self,
        name_table: Optional[Mapping[str, MetricEntry]] = None,
        override_table: Optional[Mapping[Tuple[str, str], MetricEntry]] = None,
        perf_fields: Optional[Mapping[str, MetricEntry]] = None,
    ) -> None:
        """Build a resolver.

        Args:
            name_table: Final report metric name -> entry. Defaults to ``METRIC_NAME_SEMANTICS``.
            override_table: ``(benchmark, metric)`` -> entry. Defaults to the collision table.
            perf_fields: Perf field key -> entry. Defaults to the perf table, imported lazily so
                this module stays importable before that table exists.
        """
        self._names = METRIC_NAME_SEMANTICS if name_table is None else name_table
        self._overrides = BENCHMARK_METRIC_OVERRIDES if override_table is None else override_table
        self._perf_fields = perf_fields

    def resolve(
        self,
        benchmark_name: str,
        final_metric_name: str,
        embedded_semantic_id: Optional[str] = None,
        primary_metric_name: Optional[str] = None,
    ) -> ResolvedSemantics:
        """Resolve one final report metric name.

        Args:
            benchmark_name: Benchmark (dataset) the metric belongs to.
            final_metric_name: Final report metric name, composed by
                ``compose_final_metric_name()``.
            embedded_semantic_id: ``semantic_id`` anchor stored in the report. It is used when the
                current catalog has no declaration for ``final_metric_name``.
            primary_metric_name: The benchmark's primary metric as a final report name. Promotes
                the matching metric to ``primary`` and demotes other non-diagnostic metrics.

        Returns:
            The resolution, never ``None`` and never raising. An undeclared name degrades to the
            diagnostic fallback and carries the audit messages naming where to declare it.
        """
        # 1. Benchmark level collision override.
        entry = self._overrides.get((benchmark_name, final_metric_name))
        if entry is not None:
            semantics = _with_primary_role(entry.resolve(final_metric_name), final_metric_name, primary_metric_name)
            return ResolvedSemantics(semantics=semantics, source=SemanticsSource.BENCHMARK_OVERRIDE)

        # 2. Metric name table.
        entry = self._names.get(final_metric_name)
        if entry is not None:
            semantics = _with_primary_role(entry.resolve(final_metric_name), final_metric_name, primary_metric_name)
            return ResolvedSemantics(semantics=semantics, source=SemanticsSource.METRIC_NAME)

        # 3. Report anchor: retain a historical declaration whose name is no longer catalogued.
        if embedded_semantic_id is not None:
            baseline = SEMANTIC_BASELINES.get(embedded_semantic_id)
            if baseline is not None:
                semantics = _with_primary_role(baseline, final_metric_name, primary_metric_name)
                return ResolvedSemantics(semantics=semantics, source=SemanticsSource.REPORT_ANCHOR)

        # 4. Diagnostic fallback: the value is shown as stored and the gap is logged.
        return ResolvedSemantics(
            semantics=diagnostic_fallback(final_metric_name),
            source=SemanticsSource.DIAGNOSTIC_FALLBACK,
            audit_messages=[_undeclared_metric_message(benchmark_name, final_metric_name)],
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
            from evalscope.metrics.semantics.perf import (
                PERF_API_PATH_SEMANTICS,
                PERF_FIELD_SEMANTICS,
                PERF_SUMMARY_COLUMN_SEMANTICS,
            )
        except ImportError:
            return {}
        return {**PERF_FIELD_SEMANTICS, **PERF_API_PATH_SEMANTICS, **PERF_SUMMARY_COLUMN_SEMANTICS}


@lru_cache(maxsize=1)
def get_semantics_resolver() -> SemanticsResolver:
    """Return the process-wide resolver reading the shipped tables."""
    return SemanticsResolver()


def hydrate_report_semantics(report: 'Report') -> 'Report':
    """Fill in the metric semantics of a report read from disk.

    Metrics that already carry runtime ``semantics`` keep their resolved contract; everything else
    is resolved. The current catalog takes precedence so corrections apply to historical reports;
    a persisted ``semantic_id`` anchor is a fallback for a name no longer in that catalog.

    The benchmark's primary metric is taken from ``Report.primary_metric_name`` (new reports) or
    from ``_meta.primary_metric`` (legacy reports), which is what keeps exactly one
    ``role=primary`` metric. The deprecated ``Report.score`` is never rewritten.

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

    primary_final_name = getattr(report, 'primary_metric_name', None)
    if not primary_final_name:
        recovered = _meta_primary_metric(benchmark_name)
        if recovered is not None:
            raw_primary, aggregation = recovered
            primary_final_name = match_primary_final_name(raw_primary, [metric.name for metric in metrics], aggregation)

    semantics_by_metric: Dict[str, MetricSemantics] = {}
    for metric in metrics:
        semantics = getattr(metric, 'semantics', None)
        if semantics is None:
            resolved = resolver.resolve(
                benchmark_name,
                metric.name,
                embedded_semantic_id=getattr(metric, 'semantic_id', None),
                primary_metric_name=primary_final_name,
            )
            resolved.log_audit_messages()
            semantics = resolved.semantics
        semantics_by_metric[metric.name] = semantics

    semantics_by_metric = apply_primary_metric_roles(semantics_by_metric, primary_final_name)
    for metric in metrics:
        metric.semantics = semantics_by_metric[metric.name]
        metric.semantic_id = metric.semantics.semantic_id

    primary = next(
        (metric for metric in metrics if metric.semantics is not None and metric.semantics.role is MetricRole.PRIMARY),
        None
    )
    if primary_final_name and any(metric.name == primary_final_name for metric in metrics):
        report.primary_metric_name = primary_final_name
    else:
        report.primary_metric_name = primary.name if primary else None

    return report
