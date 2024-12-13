# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class CustomModel(ABC):

    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.kwargs = kwargs

        if config.get('model_id', None) is None:
            raise ValueError(f'**Error: model_id is required in config for CustomModel. Got config: {config}')

    @abstractmethod
    @torch.no_grad()
    def predict(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Model prediction function for batch inputs.

        Args:
            prompts (str): The input batch of prompts to predict.

            **kwargs: kwargs

        Returns:
            res (dict): The model prediction results (batch). Format:
            [
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
            ,
            ...
            ]
        """
        raise NotImplementedError
