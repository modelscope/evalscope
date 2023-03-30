# Copyright (c) Alibaba, Inc. and its affiliates.

from evals.constants import PredictorMode
from evals.predictors.base import Predictor


class GptPredictor(Predictor):
    """
    OpenAI GPT models predictor, including GPT-3.5-Turbo, GPT-4, ...
    """

    def __init__(self, mode=PredictorMode.REMOTE, **kwargs):
        super(GptPredictor, self).__init__(mode=mode, **kwargs)

    def predict(self, **kwargs) -> dict:
        if self.mode == PredictorMode.REMOTE:
            result = self._run_remote_inference(**kwargs)
        elif self.mode == PredictorMode.LOCAL:
            raise ValueError(f'GPT predictor does not support local inference')
        else:
            raise ValueError(f"Invalid predictor mode: {self.mode}")

        return result

    def _run_local_inference(self, **kwargs):
        """
        Note: GPT predictor does not support local inference, no need to implement this method.
        """
        pass

    def _run_remote_inference(self, **kwargs):
        # TODO: to be implemented
        pass


