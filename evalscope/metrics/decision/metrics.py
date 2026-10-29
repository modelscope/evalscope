"""结构化决策的单样本概率指标。"""

import json
import math

from evalscope.api.metric import Metric
from evalscope.api.metric.decision import NLL_EPSILON
from evalscope.api.registry import register_metric


class _ProbabilityMetric(Metric):
    operation = ''

    def apply(self, predictions: list[str], references: list[str]) -> list[float]:
        results = []
        for prediction, reference in zip(predictions, references):
            p = json.loads(prediction)
            y = json.loads(reference)
            if self.operation == 'nll':
                value = -math.log(max(p[int(y)], NLL_EPSILON))
            elif self.operation == 'brier':
                value = sum((v - float(i == int(y))) ** 2 for i, v in enumerate(p))
            elif self.operation == 'binary_brier':
                value = (p[1] - float(y)) ** 2
            elif self.operation == 'rps':
                value = sum((sum(p[: i + 1]) - float(int(y) <= i)) ** 2 for i in range(len(p) - 1)) / (len(p) - 1)
            else:
                value = abs(sum(i * v for i, v in enumerate(p)) - float(y))
            results.append(value)
        return results


@register_metric('decision_nll')
class DecisionNLL(_ProbabilityMetric):
    """使用明确的 epsilon 截断概率后计算负对数似然。"""

    operation = 'nll'


@register_metric('decision_brier')
class DecisionBrier(_ProbabilityMetric):
    """计算所有类别误差平方和的多分类 Brier 分数。"""

    operation = 'brier'


@register_metric('decision_binary_brier')
class DecisionBinaryBrier(_ProbabilityMetric):
    """计算 P(yes) 的二元 Brier 分数。"""

    operation = 'binary_brier'


@register_metric('decision_rps')
class DecisionRPS(_ProbabilityMetric):
    """计算按等级边界数量归一化的 Ranked Probability Score。"""

    operation = 'rps'


@register_metric('decision_mae')
class DecisionMAE(_ProbabilityMetric):
    """计算有序等级期望值的绝对误差。"""

    operation = 'mae'
