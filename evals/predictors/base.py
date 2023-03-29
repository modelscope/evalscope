# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod

from evals.constants import PredictorMode


class Predictor(ABC):
    # TODO: support loading model from disk and calling remote service

    def __init__(self, mode=PredictorMode.REMOTE, **kwargs):
        self.mode = mode

    def __call__(self, **kwargs):
        return self.predict(**kwargs)

    @abstractmethod
    def predict(self, **kwargs) -> dict:
        raise NotImplementedError

    def _load_model_from_disk(self, **kwargs):
        ...

    def _call_remote_service(self, **kwargs):
        ...

