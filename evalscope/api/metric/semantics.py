"""Metric semantics contract layer.

This module defines the single authoritative data contract describing how one final
report metric is interpreted and displayed (kind, direction, unit, and display rules).
It is data-free and depends on no table: the baseline table, the catalog entry model,
the legacy mapping table and the resolver all live under ``evalscope.metrics.semantics``
and import this module, never the other way round.
"""

import json
import math
import re
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Dict, FrozenSet, Optional, Tuple, Union
from typing_extensions import Self

DIAGNOSTIC_FALLBACK_SEMANTIC_ID = 'diagnostic.unspecified'
"""Semantic identifier used when a metric has no declared meaning."""

DIAGNOSTIC_FALLBACK_PRECISION = 4
"""Display precision of an undeclared diagnostic metric."""

Scalar = Union[str, int, float, bool]
_CANONICAL_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')

KNOWN_AGGREGATIONS: FrozenSet[str] = frozenset({
    # No aggregation: the value is reported as a single number.
    'identity',
    # Averages.
    'mean',
    'macro_mean',
    'micro_mean',
    'weighted_mean',
    'clipped_mean',
    # k-sample aggregations. The k itself is `dimensions.k`, never part of this name.
    'pass_at_k',
    'pass_hat_k',
    'vote_at_k',
    'max',
    # Benchmark-owned aggregations, computed inside an adapter's `aggregate_scores`.
    'official',
    'rate',
})
"""Aggregation names that may appear in a :class:`MetricIdentity`.

This is the vocabulary of the identity's aggregation axis, which is *not* the aggregator registry:
a registered aggregator writes its own ``name`` into every aggregate it produces, adapters that
override ``aggregate_scores`` pass their own, and ``migrate_legacy_identity`` rewrites a few more.
``mean_and_pass_at_k`` and friends are deliberately absent: they emit the explicit
``pass_at_k``, ``pass_hat_k`` and ``vote_at_k`` identities alongside the ordinary mean.

It is enforced by a test rather than by this model: an aggregation name can be assembled from data,
and refusing to build an identity would abort a whole run over a presentation detail. Adding an
aggregation therefore means adding it here, deliberately.
"""


def _scalar_key(value: Scalar) -> Tuple[str, Scalar]:
    """Comparable JSON scalar key that keeps booleans distinct from numbers."""
    if isinstance(value, bool):
        return 'boolean', value
    if isinstance(value, (int, float)):
        return 'number', value
    return 'string', value


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
        return normalized

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


class MetricKind(str, Enum):
    """Whether a metric grades quality or only describes a run."""

    QUALITY = 'quality'
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

    kind: MetricKind
    """Intrinsic classification. Report-level primary selection lives on ``Report``."""

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

    @classmethod
    def diagnostic(cls, metric_name: str, semantic_id: Optional[str] = None) -> Self:
        """Build the shared fallback contract for an undeclared metric."""
        return cls(
            semantic_id=semantic_id or DIAGNOSTIC_FALLBACK_SEMANTIC_ID,
            metric_name=metric_name,
            kind=MetricKind.DIAGNOSTIC,
            direction=MetricDirection.NONE,
            display_kind=MetricDisplayKind.NUMBER,
            display_precision=DIAGNOSTIC_FALLBACK_PRECISION,
        )

    @model_validator(mode='after')
    def _check_kind_direction_display(self) -> Self:
        # Quality metrics must declare an optimization direction.
        if self.kind is MetricKind.QUALITY and self.direction == MetricDirection.NONE:
            raise ValueError(
                f"semantic_id='{self.semantic_id}', metric_name='{self.metric_name}': "
                "kind='quality' requires a direction other than 'none'"
            )

        # Diagnostic metrics carry no optimization direction.
        if self.kind is MetricKind.DIAGNOSTIC and self.direction != MetricDirection.NONE:
            raise ValueError(
                f"semantic_id='{self.semantic_id}', metric_name='{self.metric_name}': "
                f"kind='diagnostic' requires direction='none', got '{self.direction.value}'"
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
