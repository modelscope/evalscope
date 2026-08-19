from .metric import Metric, SingletonMetric, T2IMetric
from .scorer import Aggregator, AggScore, JudgeSummary, SampleScore, Score, Value
from .semantics import (
    MetricDirection,
    MetricDisplayKind,
    MetricIdentity,
    MetricKind,
    MetricSelector,
    MetricSemantics,
    Scalar,
    ValueRange,
)

# NOTE: ``MetricEntry`` is deliberately not re-exported here. It resolves against the baseline
# table and therefore lives in ``evalscope.metrics.semantics.entry``; re-exporting it from the
# ``api`` package would make the contract layer depend on the data layer again.
