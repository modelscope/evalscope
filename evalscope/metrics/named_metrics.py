from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict

from evalscope.metrics.metrics import mean, pass_at_k, weighted_mean
from evalscope.metrics.t2v_metrics import (blip2_score, clip_flant5_score, clip_score, fga_blip2_score, hpsv2_1_score,
                                           hpsv2_score, image_reward_score, mps_score, pick_score)


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
        try:
            return self.metrics[name]
        except KeyError:
            raise KeyError(f'Metric {name} not found in the registry. Available metrics: {self.list_metrics()}')

    def list_metrics(self):
        return list(self.metrics.keys())


metric_registry = MetricRegistry()

# Register metrics
metric_registry.register(Metric(name='AverageAccuracy', object=mean))
metric_registry.register(Metric(name='WeightedAverageAccuracy', object=weighted_mean))
metric_registry.register(Metric(name='AverageBLEU', object=mean))
metric_registry.register(Metric(name='AverageRouge', object=mean))
metric_registry.register(Metric(name='WeightedAverageBLEU', object=weighted_mean))
metric_registry.register(Metric(name='AveragePass@1', object=mean))
for k in range(1, 17):
    metric_registry.register(Metric(name=f'Pass@{k}', object=partial(pass_at_k, k=k)))

# t2v_metrics
metric_registry.register(Metric(name='VQAScore', object=clip_flant5_score))
metric_registry.register(Metric(name='PickScore', object=pick_score))
metric_registry.register(Metric(name='CLIPScore', object=clip_score))
metric_registry.register(Metric(name='BLIPv2Score', object=blip2_score))
metric_registry.register(Metric(name='HPSv2Score', object=hpsv2_score))
metric_registry.register(Metric(name='HPSv2.1Score', object=hpsv2_1_score))
metric_registry.register(Metric(name='ImageRewardScore', object=image_reward_score))
metric_registry.register(Metric(name='FGA_BLIP2Score', object=fga_blip2_score))
metric_registry.register(Metric(name='MPS', object=mps_score))
