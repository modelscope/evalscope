"""Primary metric summary helper.

``Report`` rendering and the reports list API share the pure function in this module so that
the "report detail" and "report list" views never disagree about what the primary metric of a
report collection is.

Hard rule for v1: this helper never averages scores across benchmarks. A summary value is only
produced when the collection contains exactly one primary metric with declared semantics. Any
multi-dataset collection reports ``None`` for both ``summary_score`` and ``summary_semantics``
(requirements 6.5, 6.6, 7.2, 7.3, 7.4).
"""

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Sequence

from evalscope.api.metric.semantics import MetricSemantics


class PrimaryMetricRef(BaseModel):
    """One dataset's primary metric as seen by the summary helper."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    dataset_name: str
    """Name of the dataset the primary metric belongs to."""

    metric_name: str
    """Final report metric name. Empty when the report has no ``role=primary`` metric."""

    score: Optional[float] = Field(default=None)
    """Score of the primary metric in its native scale. ``None`` when unavailable."""

    semantics: Optional[MetricSemantics] = Field(default=None)
    """Semantics of the primary metric. ``None`` when the report carries no declaration."""


class SummaryStatus(str, Enum):
    """Whether a report collection has a single meaningful summary value."""

    SINGLE_METRIC = 'single_metric'
    """Exactly one primary metric with declared semantics: its value is the summary value."""

    NO_AGGREGATE = 'no_aggregate'
    """Several primary metrics sharing one ``semantic_id``: comparable, but never aggregated."""

    MIXED_METRICS = 'mixed_metrics'
    """Heterogeneous or undeclared primary metrics: no summary value at all."""


class MetricSummary(BaseModel):
    """Outcome of summarizing the primary metrics of a report collection."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    status: SummaryStatus
    """Summary status of the collection."""

    summary_score: Optional[float] = Field(default=None)
    """Summary value, only set for ``single_metric``. Never a cross-benchmark mean."""

    summary_semantics: Optional[MetricSemantics] = Field(default=None)
    """Semantics of ``summary_score``, only set for ``single_metric``."""

    primary_metrics: List[PrimaryMetricRef] = Field(default_factory=list)
    """The input references, one per report, in input order."""


def summarize_primary_metrics(refs: Sequence[PrimaryMetricRef]) -> MetricSummary:
    """Summarize the primary metrics of a report collection without aggregating them.

    Args:
        refs: Primary metric references, one per report, in display order.

    Returns:
        A ``MetricSummary`` whose ``summary_score`` and ``summary_semantics`` are only set when
        the collection holds exactly one primary metric with declared semantics. Several primary
        metrics sharing one ``semantic_id`` yield ``no_aggregate``; heterogeneous, undeclared or
        empty collections yield ``mixed_metrics``. No equal-weight mean is ever computed.
    """
    primary_metrics = list(refs)
    declared = [ref for ref in primary_metrics if ref.semantics is not None]
    fully_declared = len(declared) == len(primary_metrics)

    # Requirement 7.3: a single declared primary metric is the summary value itself.
    if fully_declared and len(primary_metrics) == 1:
        only = primary_metrics[0]
        return MetricSummary(
            status=SummaryStatus.SINGLE_METRIC,
            summary_score=only.score,
            summary_semantics=only.semantics,
            primary_metrics=primary_metrics,
        )

    # Requirement 7.4: homogeneous semantics stay comparable but are never merged into one score.
    semantic_ids = {ref.semantics.semantic_id for ref in declared}
    if fully_declared and len(primary_metrics) > 1 and len(semantic_ids) == 1:
        return MetricSummary(status=SummaryStatus.NO_AGGREGATE, primary_metrics=primary_metrics)

    # Requirements 6.6, 7.4: heterogeneous, undeclared or empty collections have no summary value.
    return MetricSummary(status=SummaryStatus.MIXED_METRICS, primary_metrics=primary_metrics)
