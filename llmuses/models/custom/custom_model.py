# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import Any, Union, Dict
import torch


class CustomModel(ABC):

    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.kwargs = kwargs

    @abstractmethod
    @torch.no_grad()
    def predict(self, inputs: Union[str, dict, list], **kwargs) -> Dict[str, Any]:
        """
        Model prediction function.

        Args:
            inputs (Union[str, dict, list]): The input data. Depending on the specific model.
                str: 'xxx'
                dict: {'data': [full_prompt]}
                list: ['xxx', 'yyy', 'zzz']

            **kwargs: kwargs
        """
        raise NotImplementedError
