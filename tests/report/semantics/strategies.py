"""Reusable Hypothesis strategies for the ``MetricSemantics`` contract property tests.

Consumed by ``test_semantics_model.py``. Strategies are grouped as:

* enum and scalar building blocks (roles, directions, display kinds, value ranges, units)
* valid ``MetricSemantics`` keyword bundles, always self-consistent with the contract rules
* deliberately invalid keyword bundles, one bundle per contract rule

Generators constrain the input space instead of filtering it, so shrinking stays meaningful.
"""
import string
from hypothesis import strategies as st
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.metric.semantics import MetricDirection, MetricDisplayKind, MetricRole, ValueRange

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
) -> Dict[str, Any]:
    """Generate keyword arguments that always build a valid ``MetricSemantics``.

    Args:
        draw: Hypothesis draw function.
        roles: Role strategy, defaults to the full role domain.
        display_kinds: Display kind strategy, defaults to the full display kind domain.

    Returns:
        A keyword mapping accepted by both ``MetricSemantics`` and ``MetricEntry``.
    """
    role = draw(roles if roles is not None else metric_roles())
    direction = MetricDirection.NONE if role == MetricRole.DIAGNOSTIC else draw(scored_directions())

    kwargs: Dict[str, Any] = {
        'semantic_id': draw(semantic_ids()),
        'metric_name': draw(metric_names()),
        'role': role,
        'direction': direction,
        'raw_unit': draw(st.none() | st.sampled_from(list(RAW_UNITS))),
    }
    kwargs.update(draw(display_fields(display_kinds)))
    return kwargs


@st.composite
def role_direction_kwargs(draw: st.DrawFn) -> Dict[str, Any]:
    """Generate kwargs whose only possibly invalid aspect is the ``role`` / ``direction`` pair.

    Every other field is generated so that no other contract rule can fire: a display bundle
    consistent with the percent and scaling rules.
    """
    kwargs = draw(valid_semantics_kwargs(roles=st.just(MetricRole.DIAGNOSTIC)))
    kwargs['role'] = draw(metric_roles())
    kwargs['direction'] = draw(metric_directions())
    return kwargs


def is_role_direction_consistent(role: MetricRole, direction: MetricDirection) -> bool:
    """Return whether a ``role`` / ``direction`` pair satisfies the contract."""
    if role in SCORED_ROLES:
        return direction != MetricDirection.NONE
    return direction == MetricDirection.NONE


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
