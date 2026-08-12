"""Declarative catalog entries resolved against the baseline table.

A catalog entry is a baseline reference plus optional field overrides. Materializing one needs
the baseline data, so this module lives next to that data rather than in the contract module:
``evalscope.api.metric.semantics`` stays a pure contract with no dependency on any table, and
the dependency runs one way only (contract <- data), which is why nothing here needs a lazy
import.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any, Dict, Optional
from typing_extensions import Self

from evalscope.api.metric.semantics import MetricDirection, MetricDisplayKind, MetricRole, MetricSemantics, ValueRange
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES

#: Location the baseline table is declared at, used in error messages.
BASELINE_TABLE_LOCATION = 'evalscope/metrics/semantics/baselines.py::SEMANTIC_BASELINES'

#: Fields of ``MetricEntry`` that override the referenced baseline when not ``None``.
#: Derived from the contract so a newly added ``MetricSemantics`` field is never silently
#: dropped from the override set (``contract_version`` is fixed, not overridable).
_ENTRY_OVERRIDE_FIELDS = tuple(name for name in MetricSemantics.model_fields if name != 'contract_version')

#: Fields a baseline-free entry must declare itself.
_ENTRY_REQUIRED_WITHOUT_BASELINE = ('semantic_id', 'role', 'direction')

__all__ = ['BASELINE_TABLE_LOCATION', 'MetricEntry', 'lookup_baseline']


def lookup_baseline(baseline_id: str) -> MetricSemantics:
    """Resolve a baseline identifier into its ``MetricSemantics``.

    Args:
        baseline_id: Key of the baseline entry, for example ``quality.accuracy.ratio``.

    Returns:
        The baseline semantics declaration.

    Raises:
        ValueError: If the baseline is not declared in the baseline table.
    """
    baseline = SEMANTIC_BASELINES.get(baseline_id)
    if baseline is None:
        raise ValueError(f"unknown baseline '{baseline_id}': declare it at {BASELINE_TABLE_LOCATION}")
    return baseline


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
