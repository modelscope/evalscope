from dataclasses import dataclass, field
from typing import Callable

from evalscope.metrics.metrics import mean, weighted_mean


@dataclass
class Metric:
    name: str = 'default_metric'
    object: Callable = field(default_factory=lambda: mean)


AverageAccuracy = Metric(name='AverageAccuracy', object=mean)
WeightedAverageAccuracy = Metric(name='WeightedAverageAccuracy', object=weighted_mean)
AverageBLEU = Metric(name='AverageBLEU', object=mean)
WeightedAverageBLEU = Metric(name='WeightedAverageBLEU', object=weighted_mean)
Pass1 = Metric(name='Pass@1', object=mean)
