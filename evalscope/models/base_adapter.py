import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

from evalscope.constants import EvalType
from evalscope.models.custom import CustomModel
from evalscope.models.local_model import LocalModel

if TYPE_CHECKING:
    from evalscope.config import TaskConfig


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


def initialize_model_adapter(task_cfg: 'TaskConfig', model_adapter_cls: 'BaseModelAdapter', base_model: 'LocalModel'):
    """Initialize the model adapter based on the task configuration."""
    if task_cfg.dry_run:
        from evalscope.models.model import DummyChatModel
        return DummyChatModel(model_cfg=dict())
    elif task_cfg.eval_type == EvalType.CUSTOM:
        if not isinstance(task_cfg.model, CustomModel):
            raise ValueError(f'Expected evalscope.models.custom.CustomModel, but got {type(task_cfg.model)}.')
        from evalscope.models import CustomModelAdapter
        return CustomModelAdapter(custom_model=task_cfg.model)
    elif task_cfg.eval_type == EvalType.SERVICE:
        from evalscope.models import ServerModelAdapter
        return ServerModelAdapter(
            api_url=task_cfg.api_url, model_id=task_cfg.model, api_key=task_cfg.api_key, seed=task_cfg.seed)
    else:
        return model_adapter_cls(
            model=base_model, generation_config=task_cfg.generation_config, chat_template=task_cfg.chat_template)
