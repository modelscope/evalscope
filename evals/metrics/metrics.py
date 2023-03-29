# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Callable, Dict, List, Optional, Union


class Metrics:
    # TODO:
    #   1. add more metrics: accuracy, precision, recall, f1, auc, mae, mse, rmse, bleu, rouge, etc.
    #   2. lazyload tobe added

    """
    Metrics.

    Examples:
        >>> from evals.metrics import Metrics
        >>> Metrics.show_all_metrics()
        >>> metrics = Metrics.get_metric('accuracy')
        >>> kwargs = {'references': [0, 1, 2, 0, 1, 2], 'predictions': [0, 1, 1, 2, 1, 2]}
        >>> results = metrics.compute(**kwargs)
        >>> print(results)

    """

    METRICS_NAME = ['accuracy', 'bleu', 'rouge']

    def __init__(self, metric_name: str = 'accuracy'):
        self._metric_name = metric_name
        self._metric_fn: Callable = None

        if self._metric_name == 'accuracy':
            # todo
            self._metric_fn = self._get_accuracy_fn()
        elif self._metric_name == 'bleu':
            self._metric_fn = self._get_bleu_fn()
        elif self._metric_name == 'rouge':
            self._metric_fn = self._get_rouge_fn()
        else:
            raise ValueError(f'Unknown metric name: {self._metric_name}')

    def _get_accuracy_fn(self):

        pass

    def _get_bleu_fn(self):
        pass

    def _get_rouge_fn(self):
        pass

    def compute(self, *kwargs):
        """
        Compute metric.
        """
        return self._metric_fn(*kwargs)

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

    # from bleu import list_bleu
    from bleu import multi_list_bleu

    ref = ['it is a white cat .', 'wow , this dog is huge .']
    ref1 = ['This cat is white .', 'wow , this is a huge dog .']
    hyp = ['it is a white kitten .', 'wowww , the dog is huge !']
    hyp1 = ["it 's a white kitten .", 'wow , this dog is huge !']

    print(multi_list_bleu([ref, ref1], [hyp, hyp1]))


    print('aaa')
