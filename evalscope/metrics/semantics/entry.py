"""Declarative catalog entries resolved against the baseline table.

A catalog entry is a baseline reference plus optional field overrides. Materializing one needs
the baseline data, so this module lives next to that data rather than in the contract module:
``evalscope.api.metric.semantics`` stays a pure contract with no dependency on any table, and
the dependency runs one way only (contract <- data), which is why nothing here needs a lazy
import.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

from evalscope.api.metric.semantics import MetricDirection, MetricSemantics, ValueRange
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES

#: Location the baseline table is declared at, used in error messages.
BASELINE_TABLE_LOCATION = 'evalscope/metrics/semantics/baselines.py::SEMANTIC_BASELINES'

#: Supported differences from a named baseline. Semantic identity, kind and display kind belong
#: to the baseline vocabulary rather than to each catalog entry.
_ENTRY_OVERRIDE_FIELDS = (
    'metric_name',
    'display_name',
    'direction',
    'raw_unit',
    'value_range',
    'display_multiplier',
    'display_unit',
    'display_precision',
)

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

    baseline: str
    """Key into the baseline table."""

    metric_name: Optional[str] = None
    display_name: Optional[str] = None
    direction: Optional[MetricDirection] = None
    raw_unit: Optional[str] = None
    value_range: Optional[ValueRange] = None
    display_multiplier: Optional[float] = None
    display_unit: Optional[str] = None
    display_precision: Optional[int] = None

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
        fields: Dict[str, Any] = lookup_baseline(self.baseline).model_dump()

        for name in _ENTRY_OVERRIDE_FIELDS:
            value = getattr(self, name)
            if value is not None:
                fields[name] = value

        fields.setdefault('metric_name', final_metric_name)
        return MetricSemantics(**fields)
