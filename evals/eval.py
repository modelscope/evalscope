# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod

from evals.model import ModelMeta


class Eval(ABC):

    def __init__(self, eval_name: str, model_meta: ModelMeta, predicted_samples: str, **kwargs):
        self._eval_name = eval_name
        self._model_meta = model_meta
        self._predicted_samples = predicted_samples
        self.kwargs = kwargs

    @property
    def eval_name(self):
        return self._eval_name

    @property
    def model_meta(self) -> ModelMeta:
        return self._model_meta

    @model_meta.setter
    def model_meta(self, model_meta):
        self._model_meta = model_meta

    @property
    def predicted_samples(self):
        return self._predicted_samples

    @predicted_samples.setter
    def predicted_samples(self, predicted_samples):
        self._predicted_samples = predicted_samples

    def get_predicted_samples(self):
        # TODO: implement this method
        return self._predicted_samples

    def get_metrics(self):
        # TODO: implement this method
        ...

    def eval_single_sample(self):
        # TODO: implement this method
        pass

    @abstractmethod
    def eval_all_samples(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
