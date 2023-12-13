# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import Any


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
