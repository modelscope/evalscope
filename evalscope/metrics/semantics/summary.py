"""Primary metric references of a report collection.

``Report`` rendering and the reports list API share this model so that the "report detail" and
"report list" views describe a report's primary metric the same way.

Hard rule for v1: nothing here aggregates scores across benchmarks. A collection is presented as
one reference per dataset -- ``dataset -> metric -> score``, each in its own native scale -- and
never collapsed into a single cross-benchmark number, which would average incomparable units.
"""

from pydantic import BaseModel, ConfigDict

from evalscope.api.metric.semantics import MetricIdentity, MetricSemantics

__all__ = ['PrimaryMetricRef']


class PrimaryMetricRef(BaseModel):
    """One dataset's primary metric, as presented by the reports API."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    dataset_name: str
    """Name of the dataset the primary metric belongs to."""

    dataset_pretty_name: str = ''
    """Human-readable dataset label; empty means callers display ``dataset_name``."""

    identity: MetricIdentity
    """Canonical metric identity."""

    score: float
    """Score of the primary metric in its native scale."""

    semantics: MetricSemantics
    """Persisted semantics of the primary metric."""
