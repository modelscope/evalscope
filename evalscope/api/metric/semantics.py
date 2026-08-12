"""Metric semantics contract layer.

This module defines the single authoritative data contract describing how one final
report metric is interpreted and displayed (direction, unit, display rules, role).
It is intentionally data-free: the semantics catalog, the legacy mapping table and
the resolver live under ``evalscope.metrics.semantics``.
"""

import json
import math
import re
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Any, Dict, Optional, Tuple, Union
from typing_extensions import Self

METRIC_CONTRACT_VERSION = 1
"""Version of the MetricSemantics contract. Bump when the contract shape changes."""

Scalar = Union[str, int, float, bool]
_CANONICAL_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')


def _scalar_key(value: Scalar) -> Tuple[str, Scalar]:
    """Comparable JSON scalar key that keeps booleans distinct from numbers."""
    if isinstance(value, bool):
        return 'boolean', value
    if isinstance(value, (int, float)):
        return 'number', value
    return 'string', value


class _FrozenDimensions(dict):
    """JSON-dict representation whose contents cannot mutate after validation."""

    @staticmethod
    def _reject_mutation(*args, **kwargs) -> None:
        raise TypeError('metric identity dimensions are immutable')

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __ior__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation

    def __hash__(self) -> int:
        return hash(tuple(self.items()))

    def __copy__(self) -> '_FrozenDimensions':
        return self

    def __deepcopy__(self, memo: Dict[int, Any]) -> '_FrozenDimensions':
        return self


def _validate_canonical_name(value: str, field_name: str) -> str:
    if not _CANONICAL_NAME_PATTERN.fullmatch(value):
        raise ValueError(f'{field_name} must be lower-case snake_case, got {value!r}')
    return value


class MetricIdentity(BaseModel):
    """Stable identity of one aggregated metric.

    Aggregation and dimensions are deliberately separate from ``name`` so changing how a
    metric is summarized never creates another metric vocabulary entry.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str
    aggregation: str
    dimensions: Dict[str, Scalar] = Field(default_factory=dict)

    @field_validator('name', 'aggregation')
    @classmethod
    def _validate_names(cls, value: str, info) -> str:
        canonical = _validate_canonical_name(value, info.field_name)
        if info.field_name == 'name' and canonical in {'score', 'overall', 'total_score'}:
            raise ValueError(f'ambiguous metric name {canonical!r} is forbidden; declare the measured concept')
        return canonical

    @field_validator('dimensions')
    @classmethod
    def _validate_dimensions(cls, dimensions: Dict[str, Scalar]) -> Dict[str, Scalar]:
        normalized: Dict[str, Scalar] = {}
        for key, value in sorted(dimensions.items()):
            _validate_canonical_name(key, 'dimension key')
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'dimension {key!r} must be a finite JSON scalar')
            if isinstance(value, float) and value.is_integer():
                # JSON and JavaScript have one numeric type. Canonicalize integral floats (and
                # negative zero) so backend and frontend identity keys stay identical.
                value = int(value)
            normalized[key] = value
        return _FrozenDimensions(normalized)

    @property
    def sort_key(self) -> Tuple[str, str, Tuple[Tuple[str, Tuple[str, Scalar]], ...]]:
        """Deterministic comparison key independent of input dictionary order."""
        return self.name, self.aggregation, tuple((key, _scalar_key(value)) for key, value in self.dimensions.items())

    def __eq__(self, other: object) -> bool:
        """Compare identities without Python's bool/int scalar coercion."""
        if not isinstance(other, MetricIdentity):
            return NotImplemented
        return self.sort_key == other.sort_key

    def __hash__(self) -> int:
        """Hash the frozen identity through its normalized scalar dimensions."""
        return hash(self.sort_key)

    @property
    def key(self) -> str:
        """Human-readable, lossless identity key for logs and table grouping."""
        dimensions = ','.join(
            f'{key}={json.dumps(value, ensure_ascii=False, separators=(",", ":"))}'
            for key, value in self.dimensions.items()
        )
        suffix = f'[{dimensions}]' if dimensions else ''
        return f'{self.name}:{self.aggregation}{suffix}'


class MetricSelector(BaseModel):
    """Partial identity used to select one metric from a report.

    Omitted aggregation and dimensions are wildcards. Supplied dimensions are matched as a
    subset, allowing a selector to constrain only the axes relevant to the benchmark headline.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str
    aggregation: Optional[str] = None
    dimensions: Dict[str, Scalar] = Field(default_factory=dict)

    @field_validator('name')
    @classmethod
    def _validate_name(cls, value: str) -> str:
        canonical = _validate_canonical_name(value, 'name')
        if canonical in {'score', 'overall', 'total_score'}:
            raise ValueError(f'ambiguous metric name {canonical!r} is forbidden; declare the measured concept')
        return canonical

    @field_validator('aggregation')
    @classmethod
    def _validate_aggregation(cls, value: Optional[str]) -> Optional[str]:
        return _validate_canonical_name(value, 'aggregation') if value is not None else None

    @field_validator('dimensions')
    @classmethod
    def _validate_dimensions(cls, dimensions: Dict[str, Scalar]) -> Dict[str, Scalar]:
        return MetricIdentity(name='placeholder', aggregation='identity', dimensions=dimensions).dimensions

    def matches(self, identity: MetricIdentity) -> bool:
        """Whether ``identity`` satisfies every constraint in this selector."""
        if self.name != identity.name:
            return False
        if self.aggregation is not None and self.aggregation != identity.aggregation:
            return False
        return all(
            key in identity.dimensions and _scalar_key(identity.dimensions[key]) == _scalar_key(value)
            for key, value in self.dimensions.items()
        )


class MetricRole(str, Enum):
    """Display tier of a metric and whether it may take part in verdicts."""

    PRIMARY = 'primary'
    AUXILIARY = 'auxiliary'
    DIAGNOSTIC = 'diagnostic'


class MetricDirection(str, Enum):
    """Optimization direction of a metric."""

    HIGHER_IS_BETTER = 'higher_is_better'
    LOWER_IS_BETTER = 'lower_is_better'
    NONE = 'none'


class MetricDisplayKind(str, Enum):
    """How a metric value is rendered."""

    NUMBER = 'number'
    PERCENT = 'percent'


class ValueRange(BaseModel):
    """Closed value range of a bounded metric."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    min: float
    max: float

    @model_validator(mode='after')
    def _check_bounds(self) -> Self:
        if not math.isfinite(self.min) or not math.isfinite(self.max) or self.min >= self.max:
            raise ValueError(f'value_range must be finite with min < max, got min={self.min}, max={self.max}')
        return self


class MetricSemantics(BaseModel):
    """Single source of truth for how one final report metric is interpreted and displayed."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    semantic_id: str
    """Unique semantic identifier, named as ``{domain}.{concept}.{unit}``."""

    metric_name: str
    """Display name of the metric. May differ from the final report metric name."""

    role: MetricRole
    """Display tier: primary / auxiliary / diagnostic."""

    direction: MetricDirection
    """Optimization direction. Diagnostic metrics must use ``none``."""

    raw_unit: Optional[str] = Field(default=None)
    """Unit of the raw stored value (``s``, ``ms``, ``tok/s``, ...)."""

    value_range: Optional[ValueRange] = Field(default=None)
    """Value range for bounded metrics. ``None`` means unbounded."""

    display_kind: MetricDisplayKind = Field(default=MetricDisplayKind.NUMBER)
    """Rendering form of the value."""

    display_multiplier: Optional[float] = Field(default=None)
    """Finite positive display multiplier. ``None`` means undeclared and is treated as ``1.0``.
    It supports percent scaling and declared unit conversions without changing the stored value."""

    display_unit: Optional[str] = Field(default=None)
    """Unit appended to the displayed value (``%``, ``s``, ``ms``, ...)."""

    display_precision: int = Field(default=4)
    """Number of decimals of the displayed value, with ties rounded toward positive infinity."""

    contract_version: int = Field(default=METRIC_CONTRACT_VERSION)
    """Version of the contract this declaration follows."""

    @model_validator(mode='after')
    def _check_role_direction_display(self) -> Self:
        # Scored roles must declare an optimization direction.
        if self.role in (MetricRole.PRIMARY, MetricRole.AUXILIARY) and self.direction == MetricDirection.NONE:
            raise ValueError(
                f"semantic_id='{self.semantic_id}', metric_name='{self.metric_name}': "
                f"role='{self.role.value}' requires a direction other than 'none'"
            )

        # Diagnostic metrics carry no optimization direction.
        if self.role == MetricRole.DIAGNOSTIC and self.direction != MetricDirection.NONE:
            raise ValueError(
                f"semantic_id='{self.semantic_id}', metric_name='{self.metric_name}': "
                f"role='diagnostic' requires direction='none', got '{self.direction.value}'"
            )

        # Percent display needs an explicit range and multiplier.
        if self.display_kind == MetricDisplayKind.PERCENT:
            missing = [
                name for name, value in (('value_range', self.value_range),
                                         ('display_multiplier', self.display_multiplier)) if value is None
            ]
            if missing:
                raise ValueError(
                    f"semantic_id='{self.semantic_id}': display_kind='percent' requires "
                    f"{' and '.join(missing)}"
                )

        # Reject non-positive scaling and negative precision.
        # value_range finiteness / min < max is already guaranteed by ValueRange, so it is
        # not re-checked here (that branch was unreachable).
        if self.display_multiplier is not None and (
            not math.isfinite(self.display_multiplier) or self.display_multiplier <= 0
        ):
            raise ValueError(
                f"semantic_id='{self.semantic_id}': display_multiplier must be a finite positive "
                f'number, got {self.display_multiplier}'
            )

        if self.display_precision < 0:
            raise ValueError(
                f"semantic_id='{self.semantic_id}': display_precision must be non-negative, "
                f'got {self.display_precision}'
            )

        return self


#: Location the baseline table is declared at, used in error messages.
BASELINE_TABLE_LOCATION = 'evalscope/metrics/semantics/baselines.py::SEMANTIC_BASELINES'


def lookup_baseline(baseline_id: str) -> MetricSemantics:
    """Resolve a baseline identifier into its ``MetricSemantics``.

    The baseline table lives in ``evalscope.metrics.semantics.baselines`` and is imported
    lazily here so this contract module stays data-free and free of an import cycle. Tests
    that need a different table monkeypatch ``SEMANTIC_BASELINES`` rather than injecting a hook.

    Args:
        baseline_id: Key of the baseline entry, for example ``quality.accuracy.ratio``.

    Returns:
        The baseline semantics declaration.

    Raises:
        ValueError: If the baseline is not declared in the baseline table.
    """
    from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
    baseline = SEMANTIC_BASELINES.get(baseline_id)
    if baseline is None:
        raise ValueError(f"unknown baseline '{baseline_id}': declare it at {BASELINE_TABLE_LOCATION}")
    return baseline


#: Fields of ``MetricEntry`` that override the referenced baseline when not ``None``.
#: Derived from the contract so a newly added ``MetricSemantics`` field is never silently
#: dropped from the override set (``contract_version`` is fixed, not overridable).
_ENTRY_OVERRIDE_FIELDS = tuple(name for name in MetricSemantics.model_fields if name != 'contract_version')

#: Fields a baseline-free entry must declare itself.
_ENTRY_REQUIRED_WITHOUT_BASELINE = ('semantic_id', 'role', 'direction')


class MetricEntry(BaseModel):
    """Declarative catalog entry: a baseline reference plus optional field overrides."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    baseline: Optional[str] = Field(default=None)
    """Key into the baseline table. ``None`` means this entry is a full override."""

    semantic_id: Optional[str] = Field(default=None)
    metric_name: Optional[str] = Field(default=None)
    role: Optional[MetricRole] = Field(default=None)
    direction: Optional[MetricDirection] = Field(default=None)
    raw_unit: Optional[str] = Field(default=None)
    value_range: Optional[ValueRange] = Field(default=None)
    display_kind: Optional[MetricDisplayKind] = Field(default=None)
    display_multiplier: Optional[float] = Field(default=None)
    display_unit: Optional[str] = Field(default=None)
    display_precision: Optional[int] = Field(default=None)

    @model_validator(mode='after')
    def _check_full_override_is_complete(self) -> Self:
        if self.baseline is not None:
            return self

        missing = [name for name in _ENTRY_REQUIRED_WITHOUT_BASELINE if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"metric entry without 'baseline' must declare {', '.join(missing)}; "
                f'either set baseline (see {BASELINE_TABLE_LOCATION}) or complete the override'
            )
        return self

    def resolve(self, final_metric_name: str) -> MetricSemantics:
        """Materialize this entry into a validated ``MetricSemantics``.

        The referenced baseline provides the base values, every non-``None`` entry field
        overrides it, and ``metric_name`` falls back to the final report metric name when
        neither the entry nor the baseline declares one. Reconstructing ``MetricSemantics``
        re-runs the full contract validation.

        Args:
            final_metric_name: Final report metric name this entry is keyed by.

        Returns:
            The resolved semantics of ``final_metric_name``.

        Raises:
            ValueError: If the referenced baseline is unknown.
            pydantic.ValidationError: If the merged declaration violates the contract.
        """
        fields: Dict[str, Any] = {}
        if self.baseline is not None:
            fields.update(lookup_baseline(self.baseline).model_dump())

        for name in _ENTRY_OVERRIDE_FIELDS:
            value = getattr(self, name)
            if value is not None:
                fields[name] = value

        fields.setdefault('metric_name', final_metric_name)
        return MetricSemantics(**fields)
