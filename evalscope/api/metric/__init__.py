from .metric import Metric, SingletonMetric, T2IMetric
from .scorer import Aggregator, AggScore, SampleScore, Score, Value
from .semantics import (
    METRIC_CONTRACT_VERSION,
    MetricDirection,
    MetricDisplayKind,
    MetricIdentity,
    MetricRole,
    MetricSelector,
    MetricSemantics,
    Scalar,
    ValueRange,
)

# NOTE: ``MetricEntry`` is deliberately not re-exported here. It resolves against the baseline
# table and therefore lives in ``evalscope.metrics.semantics.entry``; re-exporting it from the
# ``api`` package would make the contract layer depend on the data layer again.
