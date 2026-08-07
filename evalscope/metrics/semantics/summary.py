"""Primary metric references of a report collection.

``Report`` rendering and the reports list API share this model so that the "report detail" and
"report list" views describe a report's primary metric the same way.

Hard rule for v1: nothing here aggregates scores across benchmarks. A collection is presented as
one reference per dataset -- ``dataset -> metric -> score``, each in its own native scale -- and
never collapsed into a single cross-benchmark number, which would average incomparable units.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

from evalscope.api.metric.semantics import MetricSemantics

__all__ = ['PrimaryMetricRef']


class PrimaryMetricRef(BaseModel):
    """One dataset's primary metric, as presented by the reports API."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    dataset_name: str
    """Name of the dataset the primary metric belongs to."""

    metric_name: str
    """Final report metric name."""

    score: Optional[float] = Field(default=None)
    """Score of the primary metric in its native scale. ``None`` when unavailable."""

    semantics: Optional[MetricSemantics] = Field(default=None)
    """Semantics of the primary metric. ``None`` when the report carries no declaration."""

    inferred: bool = Field(default=False)
    """Whether the benchmark declared this metric as primary, or it was inferred to show a value."""
