"""决策指标包，集中注册指标并保留原有模块的公开导入名称。"""

from evalscope.metrics.aggregators.decision import DecisionAggregator
from evalscope.metrics.utils.functions import auroc, average_ranks, ece

from .metrics import DecisionBinaryBrier, DecisionBrier, DecisionMAE, DecisionNLL, DecisionRPS

__all__ = [
    'DecisionBinaryBrier',
    'DecisionBrier',
    'DecisionMAE',
    'DecisionNLL',
    'DecisionRPS',
    'DecisionAggregator',
    'auroc',
    'average_ranks',
    'ece',
]
