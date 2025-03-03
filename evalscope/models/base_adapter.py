import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Union

from evalscope.constants import EvalType, OutputType
from evalscope.models.custom import CustomModel
from evalscope.models.local_model import LocalModel
from evalscope.models.register import get_model_adapter, register_model_adapter
from evalscope.utils.logger import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from evalscope.benchmarks import BenchmarkMeta
    from evalscope.config import TaskConfig


@register_model_adapter('base')
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


def initialize_model_adapter(task_cfg: 'TaskConfig', benchmark: 'BenchmarkMeta', base_model: 'LocalModel'):
    """Initialize the model adapter based on the task configuration."""
    if task_cfg.dry_run:
        from evalscope.models.model import DummyChatModel
        return DummyChatModel(model_cfg=dict())
    elif task_cfg.eval_type == EvalType.CUSTOM:
        if not isinstance(task_cfg.model, CustomModel):
            raise ValueError(f'Expected evalscope.models.custom.CustomModel, but got {type(task_cfg.model)}.')
        from evalscope.models import CustomModelAdapter
        return CustomModelAdapter(custom_model=task_cfg.model)
    elif task_cfg.eval_type == EvalType.SERVICE or task_cfg.api_url is not None:
        from evalscope.models import ServerModelAdapter

        if benchmark.model_adapter in [OutputType.CONTINUOUS, OutputType.MULTIPLE_CHOICE]:
            logger.warning('Output type is set to logits. This is not supported for service evaluation. '
                           'Setting output type to generation by default.')
            benchmark.model_adapter = OutputType.GENERATION

        return ServerModelAdapter(
            api_url=task_cfg.api_url,
            model_id=task_cfg.model,
            api_key=task_cfg.api_key,
            seed=task_cfg.seed,
            timeout=task_cfg.timeout,
            stream=task_cfg.stream,
        )
    else:
        # for local model, we need to determine the model adapter class based on the output type
        model_adapter_cls = benchmark.model_adapter
        if model_adapter_cls not in benchmark.output_types:
            logger.warning(f'Output type {model_adapter_cls} is not supported for benchmark {benchmark.name}. '
                           f'Using {benchmark.output_types[0]} instead.')
            model_adapter_cls = benchmark.output_types[0]

        model_adapter = get_model_adapter(model_adapter_cls)
        return model_adapter(
            model=base_model, generation_config=task_cfg.generation_config, chat_template=task_cfg.chat_template)
