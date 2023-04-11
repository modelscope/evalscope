# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from evals.constants import PredictorMode, PredictorEnvs, EvalTaskConfig
from evals.predictors.base import Predictor

import dashscope
from dashscope import Generation

DEFAULT_MAX_LEN = 500
DEFAULT_TOP_K = 10


class QwenPredictor(Predictor):
    # TODO:
    #   1. class name to be confirmed
    #   2. tdb

    def __init__(self, api_key: str, mode=PredictorMode.REMOTE, **kwargs):
        super(QwenPredictor, self).__init__(api_key=api_key, mode=mode, **kwargs)

        if not self.api_key:
            self.api_key = os.environ.get(PredictorEnvs.DASHSCOPE_API_KEY, None)
        if not self.api_key:
            raise ValueError(f"API key is not specified. Please set it in the environment variable "
                             f"{PredictorEnvs.DASHSCOPE_API_KEY} or pass it to the constructor.")

        self.model_name = kwargs.pop(EvalTaskConfig.ARGS_MODEL, '')
        if not self.model_name:
            raise ValueError("Model name of predictor is not specified. "
                             "Please set it in the task configuration or pass it to the constructor.")

        self.max_length = int(kwargs.pop(EvalTaskConfig.ARGS_MAX_LEN, DEFAULT_MAX_LEN))
        self.top_k = int(kwargs.pop(EvalTaskConfig.ARGS_TOP_K, DEFAULT_TOP_K))

    def predict(self, **input_kwargs) -> dict:
        """
        Run inference with the given input arguments.
        
        :param input_kwargs: input arguments for inference. Cols of kwargs:
            prompt: prompt text
            history: list of dict, each dict contains user and bot utterances
            
            Example:
                input_args = dict(
                    prompt='推荐一个附近的公园',
                    history=[
                        {
                            "user": "今天天气好吗？",
                            "bot": "今天天气不错，要出去玩玩嘛？"
                        },
                        {
                            "user": "那你有什么地方推荐？",
                            "bot": "我建议你去公园，春天来了，花朵开了，很美丽。"
                        }
                    ],
                )
            
        :return: dict, output of inference.
            Example: {'input': {}, 'output': {}}
        """

        model_info = {EvalTaskConfig.ARGS_MODEL: self.model_name,
                      EvalTaskConfig.ARGS_MAX_LEN: self.max_length,
                      EvalTaskConfig.ARGS_TOP_K: self.top_k}
        input_kwargs.update(model_info)

        if self.mode == PredictorMode.LOCAL:
            result = self._run_local_inference(**input_kwargs)
        elif self.mode == PredictorMode.REMOTE:
            result = self._run_remote_inference(**input_kwargs)
        else:
            raise ValueError(f"Invalid predictor mode: {self.mode}")

        final_res = dict(input=input_kwargs, output=result)

        return final_res

    def _run_local_inference(self, **kwargs):
        # TODO: to be implemented
        return None

    def _run_remote_inference(self, **kwargs) -> dict:
        try:
            dashscope.api_key = self.api_key
            is_debug = os.environ.get(PredictorEnvs.DEBUG_MODE)
            if is_debug == 'true':
                endpoint = os.environ.get(PredictorEnvs.DEBUG_DASHSCOPE_HTTP_BASE_URL)
                if not endpoint:
                    raise ValueError(f"Debug endpoint is not specified when DEBUG_MODE is set to true, "
                                     f"please set env: DEBUG_DASHSCOPE_HTTP_BASE_URL")
                dashscope.base_http_api_url = endpoint

            responses = Generation.call(**kwargs)
            QwenPredictor._check_response_on_error(responses)
        except Exception as e:
            raise e

        # TODO: output format to be confirmed
        return responses.output

    @staticmethod
    def _check_response_on_error(resp):
        if resp.status_code != 200:
            raise ValueError(f"Failed to call remote inference service: errCode: {resp.code}, errMsg: {resp.message}")
