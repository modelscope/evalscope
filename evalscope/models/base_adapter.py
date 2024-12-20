import torch
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from evalscope.models.custom import CustomModel
from evalscope.models.local_model import LocalModel


class BaseModelAdapter(ABC):

    def __init__(self, model: Optional[Union[LocalModel, CustomModel]], **kwargs):
        if model is None:
            self.model_cfg = kwargs.get('model_cfg', None)
        elif isinstance(model, LocalModel):
            self.model = model.model
            self.model_id = model.model_id
            self.model_revision = model.model_revision
            self.device = model.device
            self.tokenizer = model.tokenizer
            self.model_cfg = model.model_cfg
        elif isinstance(model, CustomModel):
            self.model_cfg = model.config
        else:
            raise ValueError(f'Unsupported model type: {type(model)}')

    @abstractmethod
    @torch.no_grad()
    def predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError
