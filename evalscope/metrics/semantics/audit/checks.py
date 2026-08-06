"""Audit error codes and checks.

Each check turns the read-only inventory of :mod:`.collectors` into a list of
:class:`AuditError` findings, every one naming the benchmark, the metric and the exact location
the missing declaration has to be written at:

* :func:`audit_undeclared_metrics` -- a collected metric name has no catalog entry
* :func:`audit_primary_metric_counts` -- a benchmark does not resolve to exactly one primary
  metric; this list *is* the ``BenchmarkMeta.primary_metric`` worklist
* :func:`audit_stale_primary_metric` -- a declared ``primary_metric`` never reaches the report
* :func:`audit_aggregation_groups` -- an ``aggregation_group`` mixes incompatible members
* :func:`audit_perf_fields` -- a public perf field key has no semantics entry

Two error codes of earlier revisions are gone: a dangling ``baseline`` reference is now rejected
when ``catalog.py`` is imported, and "benchmark has no catalog declaration" is meaningless once
the catalog is keyed by metric name -- undeclared metrics and primary counts cover it.
"""

import re
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evalscope.api.metric.semantics import MetricEntry, MetricRole, MetricSemantics
from evalscope.metrics.semantics.catalog import (
    BENCHMARK_DYNAMIC_METRICS,
    BENCHMARK_METRIC_OVERRIDES,
    DYNAMIC_METRIC_TABLE_LOCATION,
    METRIC_NAME_SEMANTICS,
    METRIC_NAME_TABLE_LOCATION,
)
from evalscope.metrics.semantics.naming import match_primary_final_name
from evalscope.metrics.semantics.resolver import PERF_FIELD_TABLE_LOCATION, catalog_entry_location
from evalscope.utils import get_logger
from .collectors import AUDIT_LOG_PREFIX, GROUP_DISPLAY_ORDER, MetricInventory, MetricRecord, PerfFieldRecord

logger = get_logger()

#: Where a benchmark declares its primary metric, used in audit messages.
PRIMARY_METRIC_FIELD_LOCATION = 'BenchmarkMeta.primary_metric'


class AuditErrorCode(str, Enum):
    """Audit error codes. The string value is what the output and CI logs show."""

    UNDECLARED_METRIC = 'E_UNDECLARED_METRIC'
    """A collected final report metric name has no catalog entry."""

    PRIMARY_COUNT = 'E_PRIMARY_COUNT'
    """A benchmark does not resolve to exactly one primary metric."""

    STALE_PRIMARY_METRIC = 'E_STALE_PRIMARY_METRIC'
    """A declared ``primary_metric`` does not correspond to any emitted metric name."""

    AGGREGATION_GROUP_CONFLICT = 'E_AGGREGATION_GROUP_CONFLICT'
    """An ``aggregation_group`` mixes incompatible declarations."""

    UNDECLARED_PERF_FIELD = 'E_UNDECLARED_PERF_FIELD'
    """A public perf field key has no semantics entry."""


#: Order errors are reported in, from the coarsest gap to the finest.
ERROR_CODE_ORDER: Tuple[AuditErrorCode, ...] = (
    AuditErrorCode.UNDECLARED_METRIC,
    AuditErrorCode.PRIMARY_COUNT,
    AuditErrorCode.STALE_PRIMARY_METRIC,
    AuditErrorCode.AGGREGATION_GROUP_CONFLICT,
    AuditErrorCode.UNDECLARED_PERF_FIELD,
)

#: Exit code of a run without audit errors.
EXIT_OK = 0

#: Exit code of a run with at least one audit error.
EXIT_AUDIT_ERRORS = 1

#: Fields that must agree inside one ``aggregation_group``.
AGGREGATION_GROUP_FIELDS: Tuple[str, ...] = ('raw_unit', 'value_range', 'direction')

#: Placeholder of a dynamic metric name pattern: ``{k}`` of a runtime sized family, or
#: ``<expr>`` of a name the AST could not evaluate. Only used to decide whether a *pattern*
#: is covered by a declared allow-list; resolution itself never matches names loosely.
_PLACEHOLDER_PATTERN = re.compile(r'\{[^{}]*\}|<[^<>]*>')


class AuditError(BaseModel):
    """One audit finding: what is missing, where, and how it is identified."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    code: AuditErrorCode
    """Error code of the finding."""

    message: str
    """Human readable message naming the identifiers and the location to add the entry at."""

    benchmark_name: Optional[str] = Field(default=None)
    """Benchmark the finding belongs to, ``None`` for perf fields and shared metric names."""

    metric_name: Optional[str] = Field(default=None)
    """Final report metric name, perf field key or field name of the finding."""

    location: str
    """Where the missing or conflicting declaration has to be written."""

    @property
    def sort_key(self) -> Tuple[int, str, str]:
        """Deterministic order: error code, then benchmark name, then metric name."""
        return (ERROR_CODE_ORDER.index(self.code), self.benchmark_name or '', self.metric_name or '')


class AuditReport(BaseModel):
    """Result of one audit run: the inventory plus every finding."""

    model_config = ConfigDict(extra='forbid')

    inventory: MetricInventory
    """The read-only inventory the checks ran against."""

    errors: List[AuditError] = Field(default_factory=list)
    """Findings, sorted by :attr:`AuditError.sort_key`."""

    @property
    def has_errors(self) -> bool:
        """Whether the run found at least one audit error."""
        return bool(self.errors)

    @property
    def exit_code(self) -> int:
        """Process exit code: non-zero when there is at least one error."""
        return EXIT_AUDIT_ERRORS if self.errors else EXIT_OK

    def error_counts(self) -> Dict[str, int]:
        """Number of findings per error code, only for codes that occurred."""
        counts: Dict[str, int] = {}
        for code in ERROR_CODE_ORDER:
            count = sum(1 for error in self.errors if error.code is code)
            if count:
                counts[code.value] = count
        return counts

    def errors_of(self, code: AuditErrorCode) -> List[AuditError]:
        """Return the findings of one error code, in report order."""
        return [error for error in self.errors if error.code is code]

    def to_json_dict(self) -> Dict[str, object]:
        """Render the report as the ``--json`` payload.

        Returns:
            A JSON serializable dict: the exit code, the findings and their counts, the three
            metric buckets, the public perf field keys and the coverage base.
        """
        return {
            'exit_code': self.exit_code,
            'error_count': len(self.errors),
            'error_counts': self.error_counts(),
            'errors': [error.model_dump(mode='json') for error in self.errors],
            'benchmarks': list(self.inventory.declarations),
            'coverage_base': list(self.inventory.coverage_base),
            'observed_paths': list(self.inventory.observed_paths),
            'metrics': {
                group.value: [{
                    'benchmark_name': record.benchmark_name,
                    'metric_name': record.metric_name,
                    'is_pattern': record.is_pattern,
                    'sources': list(record.sources),
                }
                              for record in self.inventory.grouped()[group]]
                for group in GROUP_DISPLAY_ORDER
            },
            'perf_field_keys': [record.model_dump(mode='json') for record in self.inventory.perf_field_keys],
        }


def _pattern_matcher(pattern: str) -> Optional[re.Pattern]:
    """Compile a dynamic metric name pattern into a matcher of concrete names.

    Args:
        pattern: Collected name containing ``{k}`` or ``<expr>`` placeholders.

    Returns:
        A matcher accepting every concrete name the pattern can stand for, or ``None`` when the
        pattern holds no placeholder and therefore has to match exactly.
    """
    placeholders = list(_PLACEHOLDER_PATTERN.finditer(pattern))
    if not placeholders:
        return None

    parts: List[str] = []
    cursor = 0
    for placeholder in placeholders:
        parts.append(re.escape(pattern[cursor:placeholder.start()]))
        parts.append('.+')
        cursor = placeholder.end()
    parts.append(re.escape(pattern[cursor:]))
    return re.compile(''.join(parts))


def _declared_names_for(
    benchmark_name: str,
    name_table: Mapping[str, MetricEntry],
    overrides: Mapping[Tuple[str, str], MetricEntry],
    dynamic: Mapping[str, Sequence[str]],
) -> List[str]:
    """Return every metric name the catalog can resolve for one benchmark."""
    names = list(name_table)
    names.extend(metric_name for (owner, metric_name) in overrides if owner == benchmark_name)
    names.extend(dynamic.get(benchmark_name, ()))
    return names


def _is_declared(record: MetricRecord, declared_names: Sequence[str]) -> bool:
    """Whether a collected record is covered by the catalog.

    A literal name has to be declared. A *pattern* stands for a family of runtime names, so it is
    covered when at least one declared name belongs to that family.

    Args:
        record: Collected record.
        declared_names: Names the catalog can resolve for the record's benchmark.

    Returns:
        ``True`` when the catalog covers the record.
    """
    if record.metric_name in declared_names:
        return True
    if not record.is_pattern:
        return False

    matcher = _pattern_matcher(record.metric_name)
    return matcher is not None and any(matcher.fullmatch(name) for name in declared_names)


def audit_undeclared_metrics(
    inventory: MetricInventory,
    name_table: Optional[Mapping[str, MetricEntry]] = None,
    overrides: Optional[Mapping[Tuple[str, str], MetricEntry]] = None,
    dynamic: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[AuditError]:
    """Report the collected metric names the catalog does not declare.

    Args:
        inventory: Inventory from ``collect_metric_inventory``.
        name_table: Metric name table. Defaults to ``METRIC_NAME_SEMANTICS``.
        overrides: Collision overrides. Defaults to ``BENCHMARK_METRIC_OVERRIDES``.
        dynamic: Dynamic allow-lists. Defaults to ``BENCHMARK_DYNAMIC_METRICS``.

    Returns:
        One ``E_UNDECLARED_METRIC`` per uncovered ``(benchmark, metric)`` pair.
    """
    resolved_names = METRIC_NAME_SEMANTICS if name_table is None else name_table
    resolved_overrides = BENCHMARK_METRIC_OVERRIDES if overrides is None else overrides
    resolved_dynamic = BENCHMARK_DYNAMIC_METRICS if dynamic is None else dynamic

    declared_by_benchmark: Dict[str, List[str]] = {}
    errors: List[AuditError] = []
    for record in inventory.records():
        if record.benchmark_name not in declared_by_benchmark:
            declared_by_benchmark[
                record.benchmark_name
            ] = _declared_names_for(record.benchmark_name, resolved_names, resolved_overrides, resolved_dynamic)
        if _is_declared(record, declared_by_benchmark[record.benchmark_name]):
            continue

        location = catalog_entry_location(record.metric_name)
        kind = 'dynamic metric name pattern' if record.is_pattern else 'metric name'
        hint = f', or list the family at {DYNAMIC_METRIC_TABLE_LOCATION}' if record.is_pattern else ''
        errors.append(
            AuditError(
                code=AuditErrorCode.UNDECLARED_METRIC,
                benchmark_name=record.benchmark_name,
                metric_name=record.metric_name,
                location=location,
                message=(
                    f"undeclared {kind}: benchmark='{record.benchmark_name}' "
                    f"metric='{record.metric_name}' group='{record.group.value}'\n"
                    f'  seen in: {"; ".join(record.sources) or "n/a"}\n'
                    f'  add an entry at {location}{hint}'
                ),
            )
        )
    return errors


def _resolved_roles(
    benchmark_name: str,
    metric_names: Sequence[str],
    primary_metric: Optional[str],
    aggregation: Optional[str],
    name_table: Mapping[str, MetricEntry],
    overrides: Mapping[Tuple[str, str], MetricEntry],
) -> Dict[str, MetricRole]:
    """Resolve the role of every emitted metric name of one benchmark.

    Mirrors the resolver: the collision override wins over the name table, and the declared
    primary metric promotes its own final name while demoting the other non-diagnostic ones.

    Args:
        benchmark_name: Benchmark the names belong to.
        metric_names: Final report metric names the benchmark emits.
        primary_metric: Raw ``BenchmarkMeta.primary_metric``, or ``None``.
        aggregation: ``BenchmarkMeta.aggregation`` of the benchmark.
        name_table: Metric name table.
        overrides: Collision overrides.

    Returns:
        Final report metric name -> resolved role, for the names that resolve.
    """
    primary_final_name = match_primary_final_name(primary_metric, metric_names, aggregation)

    roles: Dict[str, MetricRole] = {}
    for metric_name in metric_names:
        entry = overrides.get((benchmark_name, metric_name)) or name_table.get(metric_name)
        if entry is None:
            continue
        try:
            semantics = entry.resolve(metric_name)
        except ValueError as error:  # pydantic.ValidationError is a ValueError
            logger.warning(f'{AUDIT_LOG_PREFIX} unresolvable catalog entry {metric_name!r}: {error}')
            continue

        role = semantics.role
        if primary_final_name is not None and role is not MetricRole.DIAGNOSTIC:
            role = MetricRole.PRIMARY if metric_name == primary_final_name else MetricRole.AUXILIARY
        roles[metric_name] = role
    return roles


def _literal_metric_names(inventory: MetricInventory, benchmark_name: str) -> List[str]:
    """Return the literal (non-pattern) metric names one benchmark emits."""
    return [
        record.metric_name
        for record in inventory.records()
        if record.benchmark_name == benchmark_name and not record.is_pattern
    ]


def audit_primary_metric_counts(
    inventory: MetricInventory,
    name_table: Optional[Mapping[str, MetricEntry]] = None,
    overrides: Optional[Mapping[Tuple[str, str], MetricEntry]] = None,
) -> List[AuditError]:
    """Report benchmarks that do not resolve to exactly one primary metric.

    This list is the ``BenchmarkMeta.primary_metric`` worklist: a benchmark emitting several
    quality metrics without declaring a primary one resolves to more than one primary and shows
    up here together with its candidates.

    Args:
        inventory: Inventory from ``collect_metric_inventory``.
        name_table: Metric name table. Defaults to ``METRIC_NAME_SEMANTICS``.
        overrides: Collision overrides. Defaults to ``BENCHMARK_METRIC_OVERRIDES``.

    Returns:
        One ``E_PRIMARY_COUNT`` per benchmark whose primary metric count is not one.
    """
    resolved_names = METRIC_NAME_SEMANTICS if name_table is None else name_table
    resolved_overrides = BENCHMARK_METRIC_OVERRIDES if overrides is None else overrides

    errors: List[AuditError] = []
    for benchmark_name in sorted(inventory.declarations):
        declaration = inventory.declarations[benchmark_name]
        metric_names = _literal_metric_names(inventory, benchmark_name)
        if not metric_names:
            continue

        roles = _resolved_roles(
            benchmark_name,
            metric_names,
            declaration.primary_metric,
            declaration.aggregation,
            resolved_names,
            resolved_overrides,
        )
        primary_names = sorted(name for name, role in roles.items() if role is MetricRole.PRIMARY)
        if len(primary_names) == 1:
            continue

        candidates = sorted(name for name, role in roles.items() if role is not MetricRole.DIAGNOSTIC)
        errors.append(
            AuditError(
                code=AuditErrorCode.PRIMARY_COUNT,
                benchmark_name=benchmark_name,
                metric_name=declaration.primary_metric,
                location=PRIMARY_METRIC_FIELD_LOCATION,
                message=(
                    f'primary metric count is {len(primary_names)}, expected 1: '
                    f"benchmark='{benchmark_name}' primary_metric={declaration.primary_metric!r} "
                    f'resolved_primaries={primary_names}\n'
                    f'  candidates: {candidates or "none"}\n'
                    f'  declare one of them as {PRIMARY_METRIC_FIELD_LOCATION} '
                    f'in the benchmark adapter'
                ),
            )
        )
    return errors


def audit_stale_primary_metric(inventory: MetricInventory) -> List[AuditError]:
    """Report a declared ``primary_metric`` that never reaches the report.

    ``BenchmarkMeta.__post_init__`` already rejects a ``primary_metric`` outside ``metric_list``,
    so this is the second line of defence: it catches the case where the emitted final name
    differs from the raw name by more than a single aggregation prefix, leaving the declaration
    unmatched.

    Args:
        inventory: Inventory from ``collect_metric_inventory``.

    Returns:
        One ``E_STALE_PRIMARY_METRIC`` per unmatched declaration.
    """
    errors: List[AuditError] = []
    for benchmark_name in sorted(inventory.declarations):
        declaration = inventory.declarations[benchmark_name]
        if declaration.primary_metric is None:
            continue

        metric_names = _literal_metric_names(inventory, benchmark_name)
        if not metric_names:
            continue
        if match_primary_final_name(declaration.primary_metric, metric_names, declaration.aggregation) is not None:
            continue

        errors.append(
            AuditError(
                code=AuditErrorCode.STALE_PRIMARY_METRIC,
                benchmark_name=benchmark_name,
                metric_name=declaration.primary_metric,
                location=PRIMARY_METRIC_FIELD_LOCATION,
                message=(
                    f"stale primary_metric: benchmark='{benchmark_name}' "
                    f"primary_metric='{declaration.primary_metric}' does not match any emitted "
                    f'metric name {sorted(metric_names)}\n'
                    f'  fix {PRIMARY_METRIC_FIELD_LOCATION} in the benchmark adapter'
                ),
            )
        )
    return errors


def _member_field_value(semantics: MetricSemantics, field_name: str) -> object:
    """Read one consistency relevant value of an aggregation group member."""
    value = getattr(semantics, field_name)
    return value.value if isinstance(value, Enum) else value


def audit_aggregation_groups(name_table: Optional[Mapping[str, MetricEntry]] = None) -> List[AuditError]:
    """Report ``aggregation_group`` values that mix incompatible members.

    A group may only combine metrics that agree on ``raw_unit``, ``value_range`` and
    ``direction``; anything else would average incomparable numbers. v1 declares no aggregation
    group at all, so this check is expected to find nothing.

    Args:
        name_table: Metric name table. Defaults to ``METRIC_NAME_SEMANTICS``.

    Returns:
        One ``E_AGGREGATION_GROUP_CONFLICT`` per group and inconsistent field.
    """
    resolved_names = METRIC_NAME_SEMANTICS if name_table is None else name_table

    members: Dict[str, List[Tuple[str, MetricSemantics]]] = {}
    for metric_name in sorted(resolved_names):
        try:
            semantics = resolved_names[metric_name].resolve(metric_name)
        except ValueError as error:  # pydantic.ValidationError is a ValueError
            logger.warning(f'{AUDIT_LOG_PREFIX} unresolvable catalog entry {metric_name!r}: {error}')
            continue
        if semantics.aggregation_group:
            members.setdefault(semantics.aggregation_group, []).append((metric_name, semantics))

    errors: List[AuditError] = []
    for group, group_members in sorted(members.items()):
        for field_name in AGGREGATION_GROUP_FIELDS:
            distinct = []
            for _, semantics in group_members:
                value = _member_field_value(semantics, field_name)
                if value not in distinct:
                    distinct.append(value)
            if len(distinct) <= 1:
                continue

            rendered = ', '.join(
                f'{metric_name}={_member_field_value(semantics, field_name)!r}'
                for metric_name, semantics in group_members
            )
            location = catalog_entry_location(group_members[0][0])
            errors.append(
                AuditError(
                    code=AuditErrorCode.AGGREGATION_GROUP_CONFLICT,
                    metric_name=field_name,
                    location=location,
                    message=(
                        f"inconsistent aggregation_group: group='{group}' field='{field_name}' "
                        f'conflicting members: {rendered}\n'
                        f'  align the declarations at {METRIC_NAME_TABLE_LOCATION}'
                    ),
                )
            )
    return errors


def _perf_field_semantics() -> Optional[Mapping[str, MetricEntry]]:
    """Return the perf field semantics table, or ``None`` while it does not exist yet."""
    try:
        from evalscope.metrics.semantics.perf import PERF_FIELD_SEMANTICS
    except ImportError:
        logger.warning(
            f'{AUDIT_LOG_PREFIX} perf field semantics are not declared yet, '
            f'every public perf field counts as undeclared: {PERF_FIELD_TABLE_LOCATION}'
        )
        return None
    return PERF_FIELD_SEMANTICS


def audit_perf_fields(
    perf_field_keys: Sequence[PerfFieldRecord],
    perf_fields: Optional[Mapping[str, MetricEntry]] = None,
) -> List[AuditError]:
    """Report public perf field keys without a semantics entry.

    Args:
        perf_field_keys: Records from ``collect_perf_field_keys``.
        perf_fields: Perf field key -> entry. Defaults to ``PERF_FIELD_SEMANTICS``, treating its
            absence as "nothing is declared yet".

    Returns:
        One ``E_UNDECLARED_PERF_FIELD`` per undeclared public field key.
    """
    declared = _perf_field_semantics() if perf_fields is None else perf_fields
    declared = {} if declared is None else declared

    errors: List[AuditError] = []
    for record in perf_field_keys:
        if record.field_key in declared:
            continue
        errors.append(
            AuditError(
                code=AuditErrorCode.UNDECLARED_PERF_FIELD,
                metric_name=record.field_key,
                location=PERF_FIELD_TABLE_LOCATION,
                message=(
                    f"undeclared perf field: field_key='{record.field_key}' "
                    f'constant={record.holder}.{record.constant_name}\n'
                    f'  add an entry at {PERF_FIELD_TABLE_LOCATION}'
                ),
            )
        )
    return errors


def run_checks(
    inventory: MetricInventory,
    name_table: Optional[Mapping[str, MetricEntry]] = None,
    overrides: Optional[Mapping[Tuple[str, str], MetricEntry]] = None,
    dynamic: Optional[Mapping[str, Sequence[str]]] = None,
    perf_fields: Optional[Mapping[str, MetricEntry]] = None,
) -> List[AuditError]:
    """Run every check against one inventory and return the findings in report order.

    Args:
        inventory: Inventory from ``collect_metric_inventory``.
        name_table: Metric name table. Defaults to ``METRIC_NAME_SEMANTICS``.
        overrides: Collision overrides. Defaults to ``BENCHMARK_METRIC_OVERRIDES``.
        dynamic: Dynamic allow-lists. Defaults to ``BENCHMARK_DYNAMIC_METRICS``.
        perf_fields: Perf field table. Defaults to ``PERF_FIELD_SEMANTICS``.

    Returns:
        Every finding, sorted by :attr:`AuditError.sort_key`.
    """
    errors: List[AuditError] = []
    errors.extend(audit_undeclared_metrics(inventory, name_table, overrides, dynamic))
    errors.extend(audit_primary_metric_counts(inventory, name_table, overrides))
    errors.extend(audit_stale_primary_metric(inventory))
    errors.extend(audit_aggregation_groups(name_table))
    errors.extend(audit_perf_fields(inventory.perf_field_keys, perf_fields))
    return sorted(errors, key=lambda error: error.sort_key)


__all__ = [
    'AGGREGATION_GROUP_FIELDS',
    'ERROR_CODE_ORDER',
    'EXIT_AUDIT_ERRORS',
    'EXIT_OK',
    'PRIMARY_METRIC_FIELD_LOCATION',
    'AuditError',
    'AuditErrorCode',
    'AuditReport',
    'audit_aggregation_groups',
    'audit_perf_fields',
    'audit_primary_metric_counts',
    'audit_stale_primary_metric',
    'audit_undeclared_metrics',
    'run_checks',
]
