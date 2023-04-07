# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod


class Evaluate(ABC):

    def __init__(self, metrics: list, **kwargs):

        self._metrics = metrics
        self.kwargs = kwargs

    @property
    def metrics_list(self) -> list:
        return self._metrics

    def get_metrics(self):
        """
        Get metric objects from metrics_list.
        """
        # TODO: do something for self._metrics_list
        ...

    @abstractmethod
    def eval_samples(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self, predicted_samples_file: str):
        raise NotImplementedError()
