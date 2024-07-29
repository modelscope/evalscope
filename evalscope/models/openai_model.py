# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time

import openai

from evalscope.models import ChatBaseModel
from evalscope.utils.logger import get_logger

logger = get_logger()


class OpenAIModel(ChatBaseModel):
    """
    APIs of OpenAI models.
    Available models: gpt-3.5-turbo, gpt-4
    """

    MAX_RETRIES = 3

    def __init__(self, model_cfg: dict, **kwargs):
        super(OpenAIModel, self).__init__(model_cfg=model_cfg, **kwargs)

        openai_api_key = os.environ.get('OPENAI_API_KEY', None)
        self.api_key = self.model_cfg.get('api_key', openai_api_key)

        if not self.api_key:
            logger.error('OpenAI API key is not provided, please set it in environment variable OPENAI_API_KEY')
            # raise ValueError(
            #     'OpenAI API key is not provided, '
            #     'please set it in environment variable OPENAI_API_KEY')

    def predict(self, model_id: str, inputs: dict, **kwargs) -> dict:

        sys_prompt: str = inputs.get('sys_prompt', '')
        user_prompt: str = inputs.get('user_prompt', '')

        # model_id: str = kwargs.get('model_id', '')
        temperature: float = kwargs.pop('temperature', 0.2)
        max_tokens: int = kwargs.pop('max_tokens', 1024)
        mode: str = kwargs.pop('mode', 'chat.completion')

        logger.info(f'Using OpenAI model_id: {model_id}')

        res = self._predict(model_id=model_id,
                            sys_prompt=sys_prompt,
                            user_prompt=user_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            mode=mode)

        return res

    def _predict(self,
                 model_id,
                 sys_prompt,
                 user_prompt,
                 temperature,
                 max_tokens,
                 mode: str = 'chat.completion',) -> dict:

        res = {}
        openai.api_key = self.api_key

        for i in range(self.MAX_RETRIES):
            try:
                if mode == 'chat.completion':
                    resp = openai.ChatCompletion.create(
                        model=model_id,
                        messages=[{
                            'role': 'system',
                            'content': sys_prompt
                        }, {
                            'role': 'user',
                            'content': user_prompt
                        }],
                        temperature=temperature,
                        max_tokens=max_tokens)

                    if resp:
                        ans_text = resp['choices'][0]['message']['content']
                        model_id = resp['model']
                    else:
                        logger.warning(
                            f'OpenAI GPT API call failed: got empty response '
                            f'for input {sys_prompt} {user_prompt}')
                        ans_text = ''
                        model_id = ''

                    res['ans_text'] = ans_text
                    res['model_id'] = model_id
                else:
                    raise ValueError(f'Invalid mode: {mode}')

                return res

            except Exception as e:
                logger.warning(f'OpenAI API call failed: {e}')
                time.sleep(3)
        logger.error(
            f'OpenAI API call failed after {self.MAX_RETRIES} retries')
        return res
