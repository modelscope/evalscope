from dataclasses import dataclass, field
from typing import Callable, Dict

from evalscope.metrics.metrics import mean, weighted_mean


@dataclass
class Metric:
    name: str = 'default_metric'
    object: Callable = field(default_factory=lambda: mean)


class MetricRegistry:

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}

    def register(self, metric: Metric):
        self.metrics[metric.name] = metric

    def get(self, name: str) -> Metric:
        return self.metrics.get(name)

    def list_metrics(self):
        return list(self.metrics.keys())


metric_registry = MetricRegistry()

# Register metrics
metric_registry.register(Metric(name='AverageAccuracy', object=mean))
metric_registry.register(Metric(name='WeightedAverageAccuracy', object=weighted_mean))
metric_registry.register(Metric(name='AverageBLEU', object=mean))
metric_registry.register(Metric(name='WeightedAverageBLEU', object=weighted_mean))
metric_registry.register(Metric(name='Pass@1', object=mean))
