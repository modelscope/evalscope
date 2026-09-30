import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from evalscope.metrics.semantics import format_perf_value, get_semantics_resolver

#: Seconds per unit, for the time units an SLA threshold may carry.
_TIME_UNIT_SECONDS = {'s': 1.0, 'ms': 0.001}

_THRESHOLD_RE = re.compile(r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z/%]*)$')


def _scale_to_raw_unit(suffix: str, raw_unit: str, value_str: str) -> float:
    """Return the factor turning *suffix* into the metric's own unit."""
    suffix_seconds = _TIME_UNIT_SECONDS.get(suffix)
    raw_seconds = _TIME_UNIT_SECONDS.get(raw_unit)
    if suffix_seconds is None or raw_seconds is None:
        raise ValueError(f"Unit '{suffix}' does not match the unit of this metric ('{raw_unit}'): {value_str}")
    return suffix_seconds / raw_seconds


def _resolve_target(text: str, field_key: Optional[str], value_str: str) -> Tuple[float, str]:
    """Split a threshold into a value in the metric's own unit and its display text.

    A bare number is read in the unit the perf reports use for that metric, so a threshold can be
    copied off the report as-is.
    """
    match = _THRESHOLD_RE.match(text.strip())
    if match is None:
        raise ValueError(f'Invalid target value in SLA param: {value_str}')

    number = float(match.group(1))
    suffix = match.group(2)
    if field_key is None:
        if suffix:
            raise ValueError(f"Cannot read unit '{suffix}' of an unknown SLA metric: {value_str}")
        return number, f'{number:g}'

    raw_unit = get_semantics_resolver().resolve_perf_field(field_key).semantics.raw_unit or ''
    if suffix and suffix != raw_unit:
        number *= _scale_to_raw_unit(suffix, raw_unit, value_str)
    return number, format_perf_value(number, field_key)


@dataclass
class SLACriterionBase(ABC):
    target: float
    display: str = ''
    """Target rendered with its unit; empty falls back to the bare number."""

    @property
    def target_text(self) -> str:
        return self.display or f'{self.target:g}'

    def _render(self, lhs: str, expression: str) -> str:
        return f'{lhs} {expression}'.strip()

    @abstractmethod
    def validate(self, actual: float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def format_cond(self, lhs: str) -> str:
        raise NotImplementedError


@dataclass
class SLALessThan(SLACriterionBase):
    def validate(self, actual: float) -> bool:
        return actual < self.target

    def format_cond(self, lhs: str) -> str:
        return self._render(lhs, f'< {self.target_text}')

    def __str__(self):
        return f'< {self.target_text}'


@dataclass
class SLALessThanOrEqualTo(SLACriterionBase):
    def validate(self, actual: float) -> bool:
        return actual <= self.target

    def format_cond(self, lhs: str) -> str:
        return self._render(lhs, f'<= {self.target_text}')

    def __str__(self):
        return f'<= {self.target_text}'


@dataclass
class SLAGreaterThan(SLACriterionBase):
    def validate(self, actual: float) -> bool:
        return actual > self.target

    def format_cond(self, lhs: str) -> str:
        return self._render(lhs, f'> {self.target_text}')

    def __str__(self):
        return f'> {self.target_text}'


@dataclass
class SLAGreaterThanOrEqualTo(SLACriterionBase):
    def validate(self, actual: float) -> bool:
        return actual >= self.target

    def format_cond(self, lhs: str) -> str:
        return self._render(lhs, f'>= {self.target_text}')

    def __str__(self):
        return f'>= {self.target_text}'


@dataclass
class SLAMax(SLACriterionBase):
    def validate(self, actual: float) -> bool:
        return True

    def format_cond(self, lhs: str) -> str:
        return self._render(lhs, '-> max')

    def __str__(self):
        return 'max'


@dataclass
class SLAMin(SLACriterionBase):
    def validate(self, actual: float) -> bool:
        return True

    def format_cond(self, lhs: str) -> str:
        return self._render(lhs, '-> min')

    def __str__(self):
        return 'min'


SLA_CRITERIA = {
    '<=': SLALessThanOrEqualTo,
    '>=': SLAGreaterThanOrEqualTo,
    '<': SLALessThan,
    '>': SLAGreaterThan,
}

# Unicode equivalents for SLA operators
UNICODE_OPERATORS = {
    '≤': '<=',
    '≥': '>=',
    '≧': '>=',
    ' ': '',  # thin space that may appear
}


def _normalize_operator(value_str: str) -> str:
    """Normalize Unicode operators to ASCII equivalents and strip whitespace."""
    value_str = value_str.strip()
    for unicode_op, ascii_op in UNICODE_OPERATORS.items():
        if unicode_op in value_str:
            value_str = value_str.replace(unicode_op, ascii_op)
    return value_str


def create_criterion(value_str: str, field_key: Optional[str] = None) -> SLACriterionBase:
    """Build a criterion from one ``--sla-params`` value.

    Args:
        value_str: Operator plus threshold, optionally unit-suffixed (``'<=2s'``, ``'<50ms'``).
        field_key: Perf contract field key of the metric, the authority for its unit.
    """
    value_str = _normalize_operator(str(value_str))

    if value_str == 'max':
        return SLAMax(target=0.0)
    if value_str == 'min':
        return SLAMin(target=0.0)

    for op_key in sorted(SLA_CRITERIA.keys(), key=len, reverse=True):
        if value_str.startswith(op_key):
            target, display = _resolve_target(value_str[len(op_key) :], field_key, value_str)
            return SLA_CRITERIA[op_key](target, display)

    raise ValueError(f'Invalid SLA param format: {value_str}. Expected format: "<=0.02", ">=0.5", etc.')
