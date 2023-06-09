# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Callable

from evals.metrics.code_metric import compute_pass_k
from evals.metrics.math_accuracy import compute_math_accuracy_one_sample
from evals.metrics.rouge_metric import compute_rouge_score


class Metrics:
    # TODO:
    #   1. add more metrics: accuracy, precision, recall, f1, auc, mae, mse, rmse, bleu, rouge, etc.
    #   2. lazyload tobe added
    #   3. add registry
    """
    Metrics.

    Examples:
        >>> from evals.metrics.metrics import Metrics
        >>> from evals.metrics.metrics import get_metric
        >>> Metrics.show_all_metrics()
        >>> metrics = get_metric('accuracy')
        >>> kwargs = {'references': [0, 1, 2, 0, 1, 2], 'predictions': [0, 1, 1, 2, 1, 2]}
        >>> results = metrics.compute(**kwargs)
        >>> print(results)

    """

    METRICS_NAME = ['accuracy', 'bleu', 'rouge', 'math_accuracy', 'pass@k']

    def __init__(self, metric_name: str = 'accuracy'):
        self._metric_name = metric_name
        self._metric_fn: Callable = None

        if self._metric_name == 'accuracy':
            # todo: to be changed to registry
            self._metric_fn = self._get_accuracy_fn()
        elif self._metric_name == 'bleu':
            self._metric_fn = self._get_bleu_fn()
        elif self._metric_name == 'rouge':
            self._metric_fn = self._get_rouge_fn()
        elif self._metric_fn == 'math_accuracy':
            self._metric_fn = self._get_math_accuracy()
        else:
            raise ValueError(f'Unknown metric name: {self._metric_name}')

    def _get_accuracy_fn(self):

        pass

    def _get_bleu_fn(self):
        pass

    def _get_rouge_fn(self):
        return compute_rouge_score

    def _get_math_accuracy(self):
        return compute_math_accuracy_one_sample

    def _get_code_passk(self):
        return compute_pass_k

    def compute(self, **kwargs):
        """
        Common method to compute metric.
        """
        return self._metric_fn(**kwargs)

    @staticmethod
    def show_all_metrics() -> None:
        """
        Show all metrics.
        :return: None
        """
        print(Metrics.METRICS_NAME)


def get_metric(metric_name) -> Metrics:
    """
    Get metric object by name.
    :param metric_name:
    :return:
    """
    if metric_name == 'bleu':
        pass
        return Metrics
    elif metric_name == 'rouge':
        return None
    else:
        raise ValueError(f'Unknown metric name: {metric_name}')


if __name__ == '__main__':

    pass
