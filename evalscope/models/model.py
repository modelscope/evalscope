# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
from abc import ABC, abstractmethod
from typing import Any, List

from evalscope.utils.logger import get_logger

logger = get_logger()


class BaseModel(ABC):

    def __init__(self, model_cfg: dict, **kwargs):
        """
        Base model class.

        Args:
            model_cfg (dict): The model configuration. Depending on the specific model. Example:
                {'model_id': 'modelscope/Llama-2-7b-chat-ms', 'revision': 'v1.0.0'}

            **kwargs: kwargs
        """
        self.model_cfg: dict = model_cfg
        self.kwargs = kwargs

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """
        Model prediction func.
        """
        raise NotImplementedError


class ChatBaseModel(BaseModel):

    def __init__(self, model_cfg: dict, **kwargs):
        """
        Chat base model class. Depending on the specific model.

        Args:
            model_cfg (dict):
                {'model_id': 'modelscope/Llama-2-7b-chat-ms', 'revision': 'v1.0.0', 'device_map': 'auto'}

            **kwargs: kwargs
        """
        super(ChatBaseModel, self).__init__(model_cfg=model_cfg, **kwargs)

    @abstractmethod
    def predict(self, inputs: dict, **kwargs) -> dict:
        """
        Model prediction func. The inputs and outputs are compatible with OpenAI Chat Completions APIs.
        Refer to: https://platform.openai.com/docs/guides/gpt/chat-completions-api

        # TODO: follow latest OpenAI API

        Args:
            inputs (dict): The input prompts and history. Input format:
                {'messages': [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': 'Who won the world series in 2020?'},
                    {'role': 'assistant', 'content': 'The Los Angeles Dodgers won the World Series in 2020.'},
                ]
                'history': [
                    {'role': 'system', 'content': 'Hello'},
                    {'role': 'user', 'content': 'Hi'}]
                }

            kwargs (dict): Could be inference configuration. Default: None.
                cfg format: {'max_length': 1024}

        Returns: The result format:
                {
                  'choices': [
                    {
                      'index': 0,
                      'message': {
                        'content': 'The 2020 World Series was played in Texas at Globe Life Field in Arlington.',
                        'role': 'assistant'
                      }
                    }
                  ],
                  'created': 1677664795,
                  # For models on the ModelScope or HuggingFace, concat model_id and revision with "-".
                  'model': 'gpt-3.5-turbo-0613',
                  'object': 'chat.completion',
                  'usage': {
                    'completion_tokens': 17,
                    'prompt_tokens': 57,
                    'total_tokens': 74
                  }
                }
        """
        raise NotImplementedError


# TODO: Remove this class after refactoring all models
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

        res = self._predict(
            model_id=model_id,
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            mode=mode)

        return res

    def _predict(
        self,
        model_id,
        sys_prompt,
        user_prompt,
        temperature,
        max_tokens,
        mode: str = 'chat.completion',
    ) -> dict:
        import openai

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
                        logger.warning(f'OpenAI GPT API call failed: got empty response '
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
        logger.error(f'OpenAI API call failed after {self.MAX_RETRIES} retries')
        return res
