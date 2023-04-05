# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import Union


class Eval(ABC):

    def __init__(self, metrics: list, predicted_samples: Union[list, str], **kwargs):

        self._metrics = metrics
        # todo: list or path
        self._predicted_samples = predicted_samples
        self.kwargs = kwargs

    @property
    def metrics_list(self) -> list:
        return self._metrics

    @property
    def predicted_samples(self) -> Union[list, str]:
        return self._predicted_samples

    def get_predicted_samples(self):
        # TODO: implement this method
        return self._predicted_samples

    def get_metrics(self):
        """
        Get metrics from metrics_list.
        """
        # TODO: do something for self._metrics_list
        ...

    @abstractmethod
    def eval_samples(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
