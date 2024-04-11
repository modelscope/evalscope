# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import Any, Union, Dict
import torch


class CustomModel(ABC):

    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.kwargs = kwargs

        if config.get('model_id', None) is None:
            raise ValueError(f"**Error: model_id is required in config for CustomModel. Got config: {config}")

    @abstractmethod
    @torch.no_grad()
    def predict(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Model prediction function.

        Args:
            prompt (str): The input prompt for the model.

            **kwargs: kwargs

        Returns:
            res (dict): The model prediction results. Format:
            {
              'choices': [
                {
                  'index': 0,
                  'message': {
                    'content': 'xxx',
                    'role': 'assistant'
                  }
                }
              ],
              'created': 1677664795,
              'model': 'gpt-3.5-turbo-0613',   # should be model_id
              'object': 'chat.completion',
              'usage': {
                'completion_tokens': 17,
                'prompt_tokens': 57,
                'total_tokens': 74
              }
            }
        """
        raise NotImplementedError
