# Copyright (c) Alibaba, Inc. and its affiliates.

from evals.constants import PredictorMode
from evals.predictors.base import Predictor

import dashscope
from http import HTTPStatus
from dashscope import Models
from dashscope import Generation


class MossPredictor(Predictor):
    # TODO:
    #   1. class name to be confirmed
    #   2. tdb

    def __init__(self, mode=PredictorMode.REMOTE, **kwargs):
        super(MossPredictor, self).__init__(mode=mode, **kwargs)

    def predict(self, **kwargs) -> dict:
        if self.mode == PredictorMode.LOCAL:
            result = self._run_local_inference(**kwargs)
        elif self.mode == PredictorMode.REMOTE:
            result = self._run_remote_inference(**kwargs)
        else:
            raise ValueError(f"Invalid predictor mode: {self.mode}")

        return result

    def _run_local_inference(self, **kwargs):
        pass

    def _run_remote_inference(self, **kwargs) -> dict:
        try:
            responses = Generation.call(**kwargs)
        except Exception as e:
            raise e

        # TODO: output format to be confirmed
        return responses.output
