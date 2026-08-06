"""Metric semantics resolution.

``SemanticsResolver`` turns a *final report metric name* (the string ``ReportGenerator`` writes
into ``Metric.name``) into a ``MetricSemantics`` using one fixed priority chain, so a freshly
generated report, a historical report and the service APIs all agree on the direction, unit and
display rules of the same metric.

Priority chain, first hit wins:

1. ``REPORT_ANCHOR`` -- the report already stores a ``semantic_id`` anchor. Materialized from
   :data:`SEMANTIC_BASELINES` and returned without consulting the name table. A ``semantic_id``
   absent from the baseline table (renamed during catalog evolution) falls through.
2. ``BENCHMARK_OVERRIDE`` -- ``(benchmark_name, final_metric_name)`` has a collision override.
3. ``METRIC_NAME`` -- the final report metric name is declared in :data:`METRIC_NAME_SEMANTICS`.
4. ``DIAGNOSTIC_FALLBACK`` -- nothing matched: ``diagnostic.unspecified``, the raw value is kept
   as is and an audit message records where to add the missing entry.

After any hit the resolver applies the benchmark level role adjustment: the benchmark's
``primary_metric`` (the final name, supplied by the generator, ``Report.primary_metric_name`` or
``_meta``) is promoted to ``primary`` and every other non-diagnostic metric is demoted to
``auxiliary``. This adjusts only the ``role`` field, it never introduces a new lookup level.

Every lookup is an exact dictionary lookup: no regular expressions, no name normalization, no
fuzzy matching and no inference from the magnitude or the range of a value.

Blocking versus degrading
-------------------------
Resolution never raises: it always returns a ``ResolvedSemantics``. The *caller* decides what a
degradation means, driven by ``ResolvedSemantics.blocks_standard_semantics`` -- true when a
governed name (a built-in benchmark or a public perf field) degraded, in which case the caller
must not emit standard semantics. ``is_builtin_benchmark`` uses the bundled
``evalscope/benchmarks/_meta/`` entries, not ``BENCHMARK_REGISTRY`` (third-party adapters
register through the very same decorator, so registry membership cannot separate them).
"""

import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from evalscope.api.metric.semantics import MetricDirection, MetricDisplayKind, MetricEntry, MetricRole, MetricSemantics
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
from evalscope.metrics.semantics.catalog import (
    BENCHMARK_DYNAMIC_METRICS,
    BENCHMARK_METRIC_OVERRIDES,
    METRIC_NAME_SEMANTICS,
    METRIC_NAME_TABLE_LOCATION,
)
from evalscope.metrics.semantics.naming import match_primary_final_name
from evalscope.utils import get_logger

if TYPE_CHECKING:
    from evalscope.report.report import Report

logger = get_logger()

#: Prefix of every audit message emitted by this module, greppable in logs.
AUDIT_MESSAGE_PREFIX = '[metric-semantics]'

#: ``semantic_id`` used when no source of the priority chain matched.
DIAGNOSTIC_FALLBACK_SEMANTIC_ID = 'diagnostic.unspecified'

#: Decimals of a diagnostic fallback value: the raw value is shown as stored.
DIAGNOSTIC_FALLBACK_PRECISION = 4

#: Where perf field semantics are declared, used in audit messages.
PERF_FIELD_TABLE_LOCATION = 'evalscope/metrics/semantics/perf.py::PERF_FIELD_SEMANTICS'

#: Directory holding one JSON file per built-in benchmark. Resolved without importing
#: ``evalscope.utils.resource_utils`` to keep this module cheap to import.
_BUILTIN_META_DIR = Path(__file__).parents[2] / 'benchmarks' / '_meta'

__all__ = [
    'AUDIT_MESSAGE_PREFIX',
    'DIAGNOSTIC_FALLBACK_SEMANTIC_ID',
    'PERF_FIELD_TABLE_LOCATION',
    'ResolvedSemantics',
    'SemanticsResolver',
    'SemanticsSource',
    'UndeclaredMetricError',
    'builtin_benchmark_names',
    'catalog_entry_location',
    'diagnostic_fallback',
    'get_semantics_resolver',
    'hydrate_report_semantics',
    'is_builtin_benchmark',
    'is_public_perf_field',
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


class UndeclaredMetricError(ValueError):
    """A governed metric or public perf field has no semantics declaration.

    Raised by :meth:`ResolvedSemantics.raise_if_blocked` so callers producing standard semantics
    output (report generation, the perf semantics API) can stop instead of emitting a degraded
    contract.
    """


class ResolvedSemantics(BaseModel):
    """Outcome of one resolution: the semantics, its source and the audit trail."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    semantics: MetricSemantics
    """The resolved contract. Always present, even for a degradation."""

    source: SemanticsSource
    """Level of the priority chain the semantics came from."""

    strict: bool = Field(default=False)
    """Whether the resolved name is governed, i.e. a degradation is an error, not a warning."""

    audit_error: bool = Field(default=False)
    """Whether ``audit_messages`` must be logged at error level."""

    audit_messages: List[str] = Field(default_factory=list)
    """Human readable messages naming the metric and where to declare it."""

    @property
    def degraded(self) -> bool:
        """Whether the diagnostic fallback was used instead of a declared semantics."""
        return self.source is SemanticsSource.DIAGNOSTIC_FALLBACK

    @property
    def blocks_standard_semantics(self) -> bool:
        """Whether the caller must not emit standard semantics output for this metric."""
        return self.degraded and self.strict

    def log_audit_messages(self) -> None:
        """Log the audit trail: error level for governed names, warning level otherwise."""
        log = logger.error if self.audit_error else logger.warning
        for message in self.audit_messages:
            log(message)

    def raise_if_blocked(self) -> None:
        """Raise when the resolution must not reach standard semantics output.

        Raises:
            UndeclaredMetricError: If ``blocks_standard_semantics`` holds.
        """
        if not self.blocks_standard_semantics:
            return
        detail = '\n'.join(self.audit_messages
                           ) or (f"{AUDIT_MESSAGE_PREFIX} undeclared metric: metric='{self.semantics.metric_name}'")
        raise UndeclaredMetricError(detail)


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


def _dynamic_allow_list_message(
    benchmark_name: str,
    final_metric_name: str,
    allowed_names: Sequence[str],
) -> str:
    """Format the audit message of a metric outside the declared dynamic allow-list."""
    return (
        f'{AUDIT_MESSAGE_PREFIX} metric outside the dynamic allow-list: '
        f"benchmark='{benchmark_name}' metric='{final_metric_name}' "
        f'allowed={sorted(allowed_names)}\n  add an entry at {catalog_entry_location(final_metric_name)}'
    )


def _undeclared_perf_field_message(field_key: str) -> str:
    """Format the audit message of a perf field that resolved to the diagnostic fallback."""
    return (
        f"{AUDIT_MESSAGE_PREFIX} undeclared perf field: field_key='{field_key}'\n"
        f'  add an entry at {PERF_FIELD_TABLE_LOCATION}'
    )


@lru_cache(maxsize=1)
def builtin_benchmark_names() -> FrozenSet[str]:
    """Names of the benchmarks bundled with EvalScope.

    Read from the ``evalscope/benchmarks/_meta/`` files, the same coverage base the audit script
    uses. The result is cached: the bundled files do not change at runtime.

    Returns:
        Benchmark names, empty when the metadata directory is unavailable.
    """
    if not _BUILTIN_META_DIR.is_dir():
        return frozenset()
    return frozenset(path.stem for path in _BUILTIN_META_DIR.glob('*.json'))


def is_builtin_benchmark(benchmark_name: str) -> bool:
    """Whether a benchmark is governed by the catalog and must declare all its metrics.

    A benchmark counts as governed when it is bundled with EvalScope, which means an unresolved
    metric name is a gap in the catalog and must block standard semantics output; any other
    benchmark is treated as third-party and degrades instead.

    Args:
        benchmark_name: Name of the benchmark that produced the metric.

    Returns:
        ``True`` for built-in benchmarks.
    """
    return benchmark_name in builtin_benchmark_names()


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


@lru_cache(maxsize=1)
def _public_perf_field_keys() -> FrozenSet[str]:
    """Field keys of the public perf contract, reflected from the perf name constants."""
    try:
        from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics
    except ImportError:  # pragma: no cover - perf constants ship with the package
        return frozenset()

    keys = set()
    for holder in (Metrics, PercentileMetrics):
        for name, value in vars(holder).items():
            if not name.startswith('_') and isinstance(value, str):
                keys.add(value)
    return frozenset(keys)


def is_public_perf_field(field_key: str) -> bool:
    """Whether a perf field key belongs to the public perf contract.

    Args:
        field_key: Key as used by the perf archive API, e.g. ``'Avg Latency (s)'``.

    Returns:
        ``True`` when the key is one of the ``Metrics`` / ``PercentileMetrics`` constants, in
        which case a missing declaration blocks the standard semantics API.
    """
    return field_key in _public_perf_field_keys()


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
        dynamic_table: Optional[Mapping[str, Sequence[str]]] = None,
        perf_fields: Optional[Mapping[str, MetricEntry]] = None,
        builtin_benchmarks: Optional[FrozenSet[str]] = None,
    ) -> None:
        """Build a resolver.

        Args:
            name_table: Final report metric name -> entry. Defaults to ``METRIC_NAME_SEMANTICS``.
            override_table: ``(benchmark, metric)`` -> entry. Defaults to the collision table.
            dynamic_table: Benchmark -> allow-list. Defaults to ``BENCHMARK_DYNAMIC_METRICS``.
            perf_fields: Perf field key -> entry. Defaults to the perf table, imported lazily so
                this module stays importable before that table exists.
            builtin_benchmarks: Benchmark names treated as governed. Defaults to
                :func:`builtin_benchmark_names`.
        """
        self._names = METRIC_NAME_SEMANTICS if name_table is None else name_table
        self._overrides = BENCHMARK_METRIC_OVERRIDES if override_table is None else override_table
        self._dynamic = BENCHMARK_DYNAMIC_METRICS if dynamic_table is None else dynamic_table
        self._perf_fields = perf_fields
        self._builtin_benchmarks = builtin_benchmarks

    def is_strict(self, benchmark_name: str) -> bool:
        """Whether a degradation of this benchmark must block standard semantics output."""
        if self._builtin_benchmarks is not None:
            return benchmark_name in self._builtin_benchmarks
        return benchmark_name in builtin_benchmark_names()

    def resolve(
        self,
        benchmark_name: str,
        final_metric_name: str,
        embedded_semantic_id: Optional[str] = None,
        primary_metric_name: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> ResolvedSemantics:
        """Resolve one final report metric name.

        Args:
            benchmark_name: Benchmark (dataset) the metric belongs to.
            final_metric_name: Final report metric name, composed by
                ``compose_final_metric_name()``.
            embedded_semantic_id: ``semantic_id`` anchor stored in the report. When it exists in
                the baseline table the result is materialized from it (``REPORT_ANCHOR``);
                otherwise the resolver falls back to resolving by name.
            primary_metric_name: The benchmark's primary metric as a final report name. Promotes
                the matching metric to ``primary`` and demotes other non-diagnostic metrics.
            strict: Whether a degradation is an error. ``None`` infers it from :meth:`is_strict`.

        Returns:
            The resolution, never ``None`` and never raising: inspect
            ``blocks_standard_semantics`` to decide whether to stop.
        """
        is_strict = self.is_strict(benchmark_name) if strict is None else strict

        # 1. Report anchor: materialize directly from the baseline table.
        if embedded_semantic_id is not None:
            baseline = SEMANTIC_BASELINES.get(embedded_semantic_id)
            if baseline is not None:
                semantics = _with_primary_role(baseline, final_metric_name, primary_metric_name)
                return ResolvedSemantics(semantics=semantics, source=SemanticsSource.REPORT_ANCHOR, strict=is_strict)

        # 2. Benchmark level collision override.
        entry = self._overrides.get((benchmark_name, final_metric_name))
        if entry is not None:
            semantics = _with_primary_role(entry.resolve(final_metric_name), final_metric_name, primary_metric_name)
            return ResolvedSemantics(semantics=semantics, source=SemanticsSource.BENCHMARK_OVERRIDE, strict=is_strict)

        # 3. Metric name table.
        entry = self._names.get(final_metric_name)
        if entry is not None:
            semantics = _with_primary_role(entry.resolve(final_metric_name), final_metric_name, primary_metric_name)
            return ResolvedSemantics(semantics=semantics, source=SemanticsSource.METRIC_NAME, strict=is_strict)

        # 4. Diagnostic fallback, with a dynamic allow-list note when one is declared.
        audit_messages: List[str] = []
        allowed_names = self._dynamic.get(benchmark_name)
        if allowed_names is not None and final_metric_name not in allowed_names:
            audit_messages.append(_dynamic_allow_list_message(benchmark_name, final_metric_name, allowed_names))
        audit_messages.append(_undeclared_metric_message(benchmark_name, final_metric_name))
        return ResolvedSemantics(
            semantics=diagnostic_fallback(final_metric_name),
            source=SemanticsSource.DIAGNOSTIC_FALLBACK,
            strict=is_strict,
            audit_error=is_strict,
            audit_messages=audit_messages,
        )

    def resolve_perf_field(self, field_key: str, strict: Optional[bool] = None) -> ResolvedSemantics:
        """Resolve one perf field key.

        Args:
            field_key: Key of the perf field, taken from the ``Metrics`` / ``PercentileMetrics``
                constants for the perf archive API, or the stable API path of an in-report perf
                metric.
            strict: Whether a degradation is an error. ``None`` infers it from
                :func:`is_public_perf_field`, so public fields block the standard semantics API
                while third-party extension fields degrade.

        Returns:
            The resolution, never ``None`` and never raising.
        """
        is_strict = is_public_perf_field(field_key) if strict is None else strict

        entry = self._perf_field_entries().get(field_key)
        if entry is not None:
            return ResolvedSemantics(
                semantics=entry.resolve(field_key),
                source=SemanticsSource.METRIC_NAME,
                strict=is_strict,
            )

        return ResolvedSemantics(
            semantics=diagnostic_fallback(field_key),
            source=SemanticsSource.DIAGNOSTIC_FALLBACK,
            strict=is_strict,
            audit_error=is_strict,
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

    Metrics that already carry a runtime ``semantics`` (freshly generated in this process) are
    left untouched; everything else is resolved. A persisted ``semantic_id`` anchor is used as
    the ``REPORT_ANCHOR``; a legacy report without one is resolved by ``(dataset_name, name)``,
    which is what makes a legacy report render exactly like a fresh one. Resolution runs
    non-strict: a read path never fails on an undeclared metric, it degrades to diagnostic and
    logs where to declare it.

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

    for metric in metrics:
        if getattr(metric, 'semantics', None) is not None:
            continue
        resolved = resolver.resolve(
            benchmark_name,
            metric.name,
            embedded_semantic_id=getattr(metric, 'semantic_id', None),
            primary_metric_name=primary_final_name,
            strict=False,
        )
        resolved.log_audit_messages()
        metric.semantics = resolved.semantics

    primary = next(
        (metric for metric in metrics if metric.semantics is not None and metric.semantics.role is MetricRole.PRIMARY),
        None
    )
    report.primary_metric_name = primary.name if primary else None

    return report
