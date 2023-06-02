# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time

import openai
from evals.constants import PredictorMode
from evals.predictors.base import Predictor

from evals.utils.logger import get_logger

logger = get_logger()


class OpenaiGptPredictor(Predictor):
    """
    OpenAI GPT models predictor.
    Available models: gpt-3.5-turbo, gpt-4

    Examples:
        >>> from evals.predictors.openai_gpt_predictor import OpenaiGptPredictor
        >>> gpt_predictor = OpenaiGptPredictor(api_key='YOUR_API_KEY')
        >>> input_msg = dict(model='gpt-3.5-turbo',
                 sys_prompt='You are a programmer.',
                 user_prompt='Give me a example for quicksort algorithm in Python.',
                 max_tokens=1024,
                 temperature=0.2)
        >>> resp = gpt_predictor.predict(**input_msg)
        >>> print(resp)
    """

    MAX_RETRIES = 3

    def __init__(self, api_key: str = None, mode=PredictorMode.REMOTE, **kwargs):
        super().__init__(api_key=api_key, mode=mode, **kwargs)

        if not self.api_key:
            self.api_key = os.environ.get('OPENAI_API_KEY', None)
        if not self.api_key:
            logger.error(f'OpenAI API key is not provided, please set it in environment variable OPENAI_API_KEY')

    def predict(self, **kwargs) -> dict:

        if self.mode == PredictorMode.REMOTE:
            result = self._run_remote_inference(**kwargs)

        elif self.mode == PredictorMode.LOCAL:
            raise ValueError(f'GPT predictor does not support local inference')

        else:
            raise ValueError(f"Invalid predictor mode: {self.mode}")

        return result

    def _run_local_inference(self, **kwargs):
        logger.error(f'GPT predictor does not support local inference')

    def _run_remote_inference(self, model, sys_prompt, user_prompt, temperature, max_tokens, mode: str = 'chat.completion', **kwargs) -> dict:
        openai.api_key = self.api_key

        for i in range(self.MAX_RETRIES):
            try:
                if mode == 'chat.completion':
                    resp = openai.ChatCompletion.create(model=model,
                                                        messages=[{"role": "system", "content": sys_prompt},
                                                                  {"role": "user", "content": user_prompt}],
                                                        temperature=temperature,
                                                        max_tokens=max_tokens)
                else:
                    raise ValueError(f'Invalid mode: {mode}')

                # openai.openai_object.OpenAIObject
                return resp

            except Exception as e:
                logger.warning(f'OpenAI API call failed: {e}')
                time.sleep(3)
        logger.error(f'OpenAI API call failed after {self.MAX_RETRIES} retries')
        return {}
