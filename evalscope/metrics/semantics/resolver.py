"""Resolve canonical metric identities into persisted display semantics.

V2 lookups use ``MetricIdentity.name`` plus an optional ``(name, aggregation)`` override. Dynamic
axes such as ``k``, scope, threshold and category remain structured dimensions and therefore do
not expand the registry. Historical semantic anchors are consulted only while migrating old
reports. Unknown third-party metrics degrade to diagnostics without changing their values.

Primary role assignment is intentionally separate and happens once per report through
``attribute_metric_roles``. The resolver never reads a data adapter and never selects a primary.
"""

from enum import Enum
from functools import lru_cache
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evalscope.api.metric.semantics import (
    DIAGNOSTIC_FALLBACK_SEMANTIC_ID,
    MetricIdentity,
    MetricRole,
    MetricSelector,
    MetricSemantics,
)
from evalscope.metrics.semantics.catalog import (
    AGGREGATION_SEMANTICS,
    BENCHMARK_METRIC_OVERRIDES,
    METRIC_DEFINITIONS,
    METRIC_NAME_TABLE_LOCATION,
)
from evalscope.metrics.semantics.perf import PERF_SEMANTICS
from evalscope.utils import get_logger

logger = get_logger()

#: Prefix of every audit message emitted by this module, greppable in logs.
AUDIT_MESSAGE_PREFIX = '[metric-semantics]'

#: Where perf field semantics are declared, used in audit messages.
PERF_FIELD_TABLE_LOCATION = 'evalscope/metrics/semantics/perf.py::PERF_SEMANTICS'

__all__ = [
    'AUDIT_MESSAGE_PREFIX',
    'DIAGNOSTIC_FALLBACK_SEMANTIC_ID',
    'ResolvedSemantics',
    'SemanticsResolver',
    'SemanticsSource',
    'attach_perf_semantics',
    'attribute_metric_roles',
    'catalog_entry_location',
    'diagnostic_fallback',
    'get_semantics_resolver',
    'resolve_perf_semantics',
]


class SemanticsSource(str, Enum):
    """Which level of the fixed priority chain produced a resolution."""

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
    return MetricSemantics.diagnostic(metric_name)


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
    """Resolve canonical metric identities into base ``MetricSemantics`` from the shipped tables."""

    def resolve(
        self,
        benchmark_name: str,
        identity: MetricIdentity,
    ) -> ResolvedSemantics:
        """Resolve one identity without assigning its report-level role.

        Args:
            benchmark_name: Benchmark (dataset) the metric belongs to.
            identity: Canonical identity emitted by an aggregator.
        Returns:
            The resolution, never ``None`` and never raising. An undeclared name degrades to the
            diagnostic fallback and carries the audit messages naming where to declare it.
        """
        metric_name = identity.name

        # 1. Benchmark level collision override.
        entry = BENCHMARK_METRIC_OVERRIDES.get((benchmark_name, metric_name))
        if entry is not None:
            semantics = entry.resolve(metric_name)
            return ResolvedSemantics(semantics=semantics, source=SemanticsSource.BENCHMARK_OVERRIDE)

        # 2. Aggregation-specific override, then the canonical name table.
        entry = AGGREGATION_SEMANTICS.get((metric_name, identity.aggregation))
        if entry is None:
            entry = METRIC_DEFINITIONS.get(metric_name)
        if entry is not None:
            semantics = entry.resolve(metric_name)
            return ResolvedSemantics(semantics=semantics, source=SemanticsSource.METRIC_NAME)

        # 3. Diagnostic fallback: the value is shown as stored and the gap is logged.
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
        entry = PERF_SEMANTICS.get(field_key)
        if entry is not None:
            return ResolvedSemantics(semantics=entry.resolve(field_key), source=SemanticsSource.METRIC_NAME)

        return ResolvedSemantics(
            semantics=diagnostic_fallback(field_key),
            source=SemanticsSource.DIAGNOSTIC_FALLBACK,
            audit_messages=[_undeclared_perf_field_message(field_key)],
        )


@lru_cache(maxsize=1)
def get_semantics_resolver() -> SemanticsResolver:
    """Return the process-wide resolver reading the shipped tables."""
    return SemanticsResolver()


def resolve_perf_semantics(field_keys: Iterable[str]) -> Dict[str, dict]:
    """Resolve the semantics of the perf fields a service response is about to return.

    A field with no declaration degrades to a diagnostic, which renders the stored value without a
    direction or unit and logs where to declare it.

    Args:
        field_keys: Field keys present in the response. They come from perf name constants or
            stable API paths declared in ``evalscope.metrics.semantics.perf``.

    Returns:
        Field key -> serialized ``MetricSemantics``, one entry per requested key.
    """
    resolver = get_semantics_resolver()
    semantics: Dict[str, dict] = {}
    for field_key in field_keys:
        resolved = resolver.resolve_perf_field(field_key)
        resolved.log_audit_messages()
        semantics[field_key] = resolved.semantics.model_dump(mode='json')
    return semantics


def attach_perf_semantics(perf_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the complete semantics map to an embedded report perf payload."""
    payload = dict(perf_metrics)
    summary = payload.get('summary')
    if not isinstance(summary, dict):
        return payload

    field_keys = ['n_samples']
    for key in ('latency', 'ttft', 'tpot'):
        if key in summary:
            field_keys.append(key)
    throughput = summary.get('throughput')
    if isinstance(throughput, dict):
        field_keys.extend(f'throughput.{key}' for key in throughput if f'throughput.{key}' in PERF_SEMANTICS)
    usage = summary.get('usage')
    if isinstance(usage, dict):
        field_keys.extend(f'usage.{key}' for key in usage if f'usage.{key}' in PERF_SEMANTICS)
    payload['metric_semantics'] = resolve_perf_semantics(field_keys)
    return payload
