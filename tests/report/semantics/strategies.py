"""Reusable Hypothesis strategies for metric semantics property tests.

Feature: metric-semantics-governance

This module is the single generator source shared by every property test under
``tests/report/semantics/``. Strategies are grouped as:

* enum and scalar building blocks (roles, directions, display kinds, value ranges, units)
* valid ``MetricSemantics`` keyword bundles, always self-consistent with the contract rules
* deliberately invalid keyword bundles, one bundle per contract rule
* catalog level generators (``MetricEntry``, metric name tables)
* resolver level generators (benchmark names, baseline keys, declared and undeclared metric names)

Generators constrain the input space instead of filtering it, so shrinking stays meaningful.
"""
import string
from hypothesis import strategies as st
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.metric.semantics import (
    MetricDirection,
    MetricDisplayKind,
    MetricEntry,
    MetricRole,
    MetricSemantics,
    ValueRange,
)
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
from evalscope.metrics.semantics.catalog import BENCHMARK_METRIC_OVERRIDES, METRIC_NAME_SEMANTICS

#: Roles that carry an optimization direction.
SCORED_ROLES: Tuple[MetricRole, ...] = (MetricRole.PRIMARY, MetricRole.AUXILIARY)

#: Directions allowed for scored roles.
SCORED_DIRECTIONS: Tuple[MetricDirection, ...] = (
    MetricDirection.HIGHER_IS_BETTER,
    MetricDirection.LOWER_IS_BETTER,
)

#: Domains used by the ``{domain}.{concept}.{unit}`` semantic_id convention.
SEMANTIC_DOMAINS: Tuple[str, ...] = ('quality', 'perf', 'diagnostic')

#: Unit suffixes used by the semantic_id convention.
SEMANTIC_UNITS: Tuple[str, ...] = ('ratio', 'points_100', 'unbounded', 'seconds', 'milliseconds', 'items')

#: Units that may show up as ``raw_unit``.
RAW_UNITS: Tuple[str, ...] = ('s', 'ms', 'tok/s', 'req/s', 'items')

#: Units that may show up as ``display_unit``.
DISPLAY_UNITS: Tuple[str, ...] = ('%', 's', 'ms', 'tok/s', 'pts')

#: Alphabet of generated identifiers such as concepts and final report metric names.
IDENTIFIER_ALPHABET: str = string.ascii_lowercase + string.digits + '_'

#: Values that are outside every closed enum domain of the contract.
INVALID_ENUM_VALUES: Tuple[Any, ...] = ('', 'unknown', 'PRIMARY', 'up', 'ratio', 'none ', 0, True, None)


def identifiers(min_size: int = 1, max_size: int = 12) -> st.SearchStrategy[str]:
    """Generate lowercase identifiers usable as concepts or final report metric names."""
    return st.text(alphabet=IDENTIFIER_ALPHABET, min_size=min_size, max_size=max_size)


def metric_roles() -> st.SearchStrategy[MetricRole]:
    """Generate any member of the closed ``MetricRole`` domain."""
    return st.sampled_from(list(MetricRole))


def scored_roles() -> st.SearchStrategy[MetricRole]:
    """Generate roles that must declare an optimization direction."""
    return st.sampled_from(list(SCORED_ROLES))


def metric_directions() -> st.SearchStrategy[MetricDirection]:
    """Generate any member of the closed ``MetricDirection`` domain."""
    return st.sampled_from(list(MetricDirection))


def scored_directions() -> st.SearchStrategy[MetricDirection]:
    """Generate directions other than ``none``."""
    return st.sampled_from(list(SCORED_DIRECTIONS))


def metric_display_kinds() -> st.SearchStrategy[MetricDisplayKind]:
    """Generate any member of the closed ``MetricDisplayKind`` domain."""
    return st.sampled_from(list(MetricDisplayKind))


def semantic_ids() -> st.SearchStrategy[str]:
    """Generate ``{domain}.{concept}.{unit}`` semantic identifiers."""
    return st.builds(
        lambda domain, concept, unit: f'{domain}.{concept}.{unit}',
        st.sampled_from(list(SEMANTIC_DOMAINS)),
        identifiers(),
        st.sampled_from(list(SEMANTIC_UNITS)),
    )


def metric_names() -> st.SearchStrategy[str]:
    """Generate metric display names."""
    return st.text(alphabet=string.ascii_letters + string.digits + '_', min_size=1, max_size=16)


def comparison_groups() -> st.SearchStrategy[str]:
    """Generate non-empty comparison group names."""
    return st.builds(
        lambda domain, concept: f'{domain}.{concept}',
        st.sampled_from(list(SEMANTIC_DOMAINS)),
        identifiers(),
    )


@st.composite
def value_ranges(draw: st.DrawFn) -> ValueRange:
    """Generate finite value ranges with ``min < max``."""
    minimum = draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    span = draw(st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False))
    return ValueRange(min=minimum, max=minimum + span)


def display_multipliers() -> st.SearchStrategy[float]:
    """Generate finite positive display multipliers."""
    return st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)


def display_precisions() -> st.SearchStrategy[int]:
    """Generate non-negative display precisions."""
    return st.integers(min_value=0, max_value=10)


@st.composite
def display_fields(
    draw: st.DrawFn,
    display_kinds: Optional[st.SearchStrategy[MetricDisplayKind]] = None,
) -> Dict[str, Any]:
    """Generate a display field bundle that satisfies the percent and scaling rules."""
    display_kind = draw(display_kinds if display_kinds is not None else metric_display_kinds())

    if display_kind == MetricDisplayKind.PERCENT:
        value_range: Optional[ValueRange] = draw(value_ranges())
        display_multiplier: Optional[float] = draw(display_multipliers())
    else:
        value_range = draw(st.none() | value_ranges())
        display_multiplier = draw(st.none() | display_multipliers())

    return {
        'value_range': value_range,
        'display_kind': display_kind,
        'display_multiplier': display_multiplier,
        'display_unit': draw(st.none() | st.sampled_from(list(DISPLAY_UNITS))),
        'display_precision': draw(display_precisions()),
    }


@st.composite
def valid_semantics_kwargs(
    draw: st.DrawFn,
    roles: Optional[st.SearchStrategy[MetricRole]] = None,
    display_kinds: Optional[st.SearchStrategy[MetricDisplayKind]] = None,
    with_comparison_group: bool = True,
) -> Dict[str, Any]:
    """Generate keyword arguments that always build a valid ``MetricSemantics``.

    Args:
        draw: Hypothesis draw function.
        roles: Role strategy, defaults to the full role domain.
        display_kinds: Display kind strategy, defaults to the full display kind domain.
        with_comparison_group: Whether scored roles may declare a comparison group.

    Returns:
        A keyword mapping accepted by both ``MetricSemantics`` and ``MetricEntry``.
    """
    role = draw(roles if roles is not None else metric_roles())

    if role == MetricRole.DIAGNOSTIC:
        direction = MetricDirection.NONE
        comparison_group: Optional[str] = None
    else:
        direction = draw(scored_directions())
        comparison_group = draw(st.none() | comparison_groups()) if with_comparison_group else None

    kwargs: Dict[str, Any] = {
        'semantic_id': draw(semantic_ids()),
        'metric_name': draw(metric_names()),
        'role': role,
        'direction': direction,
        'raw_unit': draw(st.none() | st.sampled_from(list(RAW_UNITS))),
        'comparison_group': comparison_group,
    }
    kwargs.update(draw(display_fields(display_kinds)))
    return kwargs


def valid_semantics(
    roles: Optional[st.SearchStrategy[MetricRole]] = None,
    display_kinds: Optional[st.SearchStrategy[MetricDisplayKind]] = None,
) -> st.SearchStrategy[MetricSemantics]:
    """Generate valid ``MetricSemantics`` instances."""
    return valid_semantics_kwargs(roles=roles, display_kinds=display_kinds).map(lambda kw: MetricSemantics(**kw))


@st.composite
def role_direction_kwargs(draw: st.DrawFn) -> Dict[str, Any]:
    """Generate kwargs whose only possibly invalid aspect is the ``role`` / ``direction`` pair.

    Every other field is generated so that no other contract rule can fire: no comparison
    group, and a display bundle consistent with the percent and scaling rules.
    """
    kwargs = draw(valid_semantics_kwargs(roles=st.just(MetricRole.DIAGNOSTIC), with_comparison_group=False))
    kwargs['role'] = draw(metric_roles())
    kwargs['direction'] = draw(metric_directions())
    return kwargs


def is_role_direction_consistent(role: MetricRole, direction: MetricDirection) -> bool:
    """Return whether a ``role`` / ``direction`` pair satisfies the contract."""
    if role in SCORED_ROLES:
        return direction != MetricDirection.NONE
    return direction == MetricDirection.NONE


@st.composite
def diagnostic_with_comparison_group_kwargs(draw: st.DrawFn) -> Dict[str, Any]:
    """Generate diagnostic kwargs that violate the comparison group rule."""
    kwargs = draw(valid_semantics_kwargs(roles=st.just(MetricRole.DIAGNOSTIC), with_comparison_group=False))
    kwargs['comparison_group'] = draw(comparison_groups())
    return kwargs


@st.composite
def percent_missing_display_field_kwargs(draw: st.DrawFn) -> Tuple[Dict[str, Any], List[str]]:
    """Generate percent kwargs missing ``value_range``, ``display_multiplier`` or both.

    Returns:
        The kwargs mapping and the names of the fields that were dropped.
    """
    kwargs = draw(valid_semantics_kwargs(display_kinds=st.just(MetricDisplayKind.PERCENT)))
    dropped = draw(
        st.lists(st.sampled_from(['value_range', 'display_multiplier']), min_size=1, max_size=2, unique=True)
    )
    for field_name in dropped:
        kwargs[field_name] = None
    return kwargs, sorted(dropped)


@st.composite
def invalid_enum_kwargs(draw: st.DrawFn) -> Tuple[Dict[str, Any], str]:
    """Generate kwargs where exactly one enum field holds a value outside its domain.

    Returns:
        The kwargs mapping and the name of the field holding the out-of-domain value.
    """
    kwargs = draw(valid_semantics_kwargs())
    field_name = draw(st.sampled_from(['role', 'direction', 'display_kind']))
    kwargs[field_name] = draw(st.sampled_from(list(INVALID_ENUM_VALUES)))
    return kwargs, field_name


def full_override_metric_entries(
    roles: Optional[st.SearchStrategy[MetricRole]] = None,
    display_kinds: Optional[st.SearchStrategy[MetricDisplayKind]] = None,
) -> st.SearchStrategy[MetricEntry]:
    """Generate baseline-free ``MetricEntry`` instances that resolve without a baseline table."""
    return valid_semantics_kwargs(roles=roles, display_kinds=display_kinds).map(lambda kw: MetricEntry(**kw))


def final_metric_names() -> st.SearchStrategy[str]:
    """Generate final report metric names used as catalog keys."""
    return identifiers(min_size=1, max_size=20)


@st.composite
def metric_name_tables(draw: st.DrawFn, max_metrics: int = 4) -> Dict[str, MetricEntry]:
    """Generate a ``METRIC_NAME_SEMANTICS`` style table of resolvable entries."""
    names = draw(st.lists(final_metric_names(), min_size=1, max_size=max_metrics, unique=True))
    roles = st.sampled_from([MetricRole.PRIMARY, MetricRole.AUXILIARY, MetricRole.DIAGNOSTIC])
    return {name: draw(full_override_metric_entries(roles=roles)) for name in names}


# ---------------------------------------------------------------------------------------------
# Resolver level generators: benchmark names and baseline keys.
# ---------------------------------------------------------------------------------------------

#: Baseline keys of the shipped baseline table, usable as ``MetricEntry.baseline``.
BASELINE_IDS: Tuple[str, ...] = tuple(sorted(SEMANTIC_BASELINES))

#: Metric names declared by the shipped catalog, i.e. names the name level may resolve.
DECLARED_METRIC_NAMES: Tuple[str, ...] = tuple(
    sorted(set(METRIC_NAME_SEMANTICS) | {metric_name
                                        for _, metric_name in BENCHMARK_METRIC_OVERRIDES})
)

#: Benchmark names carrying a benchmark level collision override.
OVERRIDE_BENCHMARK_NAMES: Tuple[str, ...] = tuple(
    sorted({benchmark_name
            for benchmark_name, _ in BENCHMARK_METRIC_OVERRIDES})
)

#: Prefix that keeps a generated benchmark name out of every shipped table.
SYNTHETIC_BENCHMARK_PREFIX: str = 'bench_'

#: Prefix that keeps a generated metric name out of every shipped table, so the only source
#: that can resolve it is the one the test installs itself.
UNDECLARED_METRIC_PREFIX: str = 'undeclared_'


def baseline_ids() -> st.SearchStrategy[str]:
    """Generate keys of the shipped baseline table."""
    return st.sampled_from(list(BASELINE_IDS))


def baseline_metric_entries() -> st.SearchStrategy[MetricEntry]:
    """Generate ``MetricEntry`` instances that only reference a baseline."""
    return baseline_ids().map(lambda baseline: MetricEntry(baseline=baseline))


@st.composite
def baseline_override_metric_entries(draw: st.DrawFn) -> MetricEntry:
    """Generate baseline references that also override at least one display field.

    Only fields that cannot invalidate the merged declaration are overridden, so the entry
    always resolves and the referenced baseline keeps deciding the role and the direction.
    """
    overrides: Dict[str, Any] = {
        'metric_name': draw(st.none() | metric_names()),
        'display_precision': draw(st.none() | display_precisions()),
        'raw_unit': draw(st.none() | st.sampled_from(list(RAW_UNITS))),
    }
    declared = {name: value for name, value in overrides.items() if value is not None}
    if not declared:
        declared = {'display_precision': draw(display_precisions())}
    return MetricEntry(baseline=draw(baseline_ids()), **declared)


def synthetic_benchmark_names() -> st.SearchStrategy[str]:
    """Generate benchmark names absent from every shipped table."""
    return identifiers().map(lambda suffix: f'{SYNTHETIC_BENCHMARK_PREFIX}{suffix}')


def benchmark_names() -> st.SearchStrategy[str]:
    """Generate benchmark names: carrying a collision override, or absent from every table."""
    if OVERRIDE_BENCHMARK_NAMES:
        return st.one_of(st.sampled_from(list(OVERRIDE_BENCHMARK_NAMES)), synthetic_benchmark_names())
    return synthetic_benchmark_names()


def declared_metric_names() -> st.SearchStrategy[str]:
    """Generate metric names the shipped catalog declares."""
    return st.sampled_from(list(DECLARED_METRIC_NAMES))


def undeclared_metric_names() -> st.SearchStrategy[str]:
    """Generate final report metric names no shipped table declares."""
    return identifiers().map(lambda suffix: f'{UNDECLARED_METRIC_PREFIX}{suffix}')


def name_variants(declared_name: str) -> st.SearchStrategy[str]:
    """Generate variants of a declared name that are themselves not declared.

    Covers the shapes a name-inference implementation would wrongly accept: case changes,
    removed or added underscores, and added prefixes / suffixes.

    Args:
        declared_name: Name that is declared in the catalog.

    Returns:
        A strategy over strings that differ from every declared name.
    """
    variants = [
        declared_name.upper(),
        declared_name.lower() + '_v2',
        declared_name.replace('_', ''),
        declared_name.replace('_', '-'),
        f' {declared_name}',
        f'{declared_name} ',
        f'x_{declared_name}',
        f'{declared_name}_x',
    ]
    return st.sampled_from([variant for variant in variants if variant not in DECLARED_METRIC_NAMES])


def metric_scores() -> st.SearchStrategy[float]:
    """Generate finite metric scores as stored in a report."""
    return st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


def metric_values() -> st.SearchStrategy[Optional[float]]:
    """Generate metric values for formatting, including the missing and non-finite paths."""
    return st.none() | st.just(float('nan')) | metric_scores()


@st.composite
def report_specs(draw: st.DrawFn, max_metrics: int = 4) -> Tuple[str, List[Tuple[str, float]]]:
    """Generate the content of one report: a benchmark name and its ``(metric, score)`` pairs.

    Args:
        draw: Hypothesis draw function.
        max_metrics: Upper bound on the number of metrics in the report.

    Returns:
        The benchmark name and the ``(final_metric_name, score)`` pairs, names unique and in
        report order.
    """
    benchmark_name = draw(benchmark_names())
    metric_names = draw(
        st.lists(st.one_of(declared_metric_names(), undeclared_metric_names()),
                 min_size=1,
                 max_size=max_metrics,
                 unique=True)
    )
    scores = draw(st.lists(metric_scores(), min_size=len(metric_names), max_size=len(metric_names)))
    return benchmark_name, list(zip(metric_names, scores))
