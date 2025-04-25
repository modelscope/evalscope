import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from evalscope.constants import DEFAULT_MODEL_CACHE_DIR, DEFAULT_MODEL_REVISION, EvalType, ModelTask
from evalscope.utils.logger import get_logger
from evalscope.utils.model_utils import get_device

if TYPE_CHECKING:
    from evalscope.config import TaskConfig

logger = get_logger()


class LocalModel(ABC):

    def __init__(self,
                 model_id: str,
                 model_revision: str = None,
                 device_map: str = None,
                 torch_dtype: str = 'auto',
                 cache_dir: str = None,
                 **kwargs):

        self.model_id = model_id
        self.model_revision = model_revision or DEFAULT_MODEL_REVISION
        self.device = device_map or get_device()
        self.cache_dir = cache_dir or DEFAULT_MODEL_CACHE_DIR
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None

        if isinstance(torch_dtype, str) and torch_dtype != 'auto':
            import torch
            torch_dtype = eval(torch_dtype)
        self.torch_dtype = torch_dtype

        self.model_cfg = {
            'model_id': self.model_id,
            'device_map': self.device,
            'torch_dtype': str(self.torch_dtype),
        }

    @abstractmethod
    def load_model(self):
        pass


class LocalChatModel(LocalModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_model(self):
        from modelscope import AutoModelForCausalLM, AutoTokenizer

        logger.info(f'Loading model {self.model_id} ...')

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

        # Fix no padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
        )


class LocalImageModel(LocalModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pipeline_cls = kwargs.pop('pipeline_cls', None)
        # default to DiffusionPipeline if not specified
        if self.pipeline_cls is None:
            if 'flux' in self.model_id.lower():
                self.pipeline_cls = 'FluxPipeline'
            else:
                self.pipeline_cls = 'DiffusionPipeline'

    def load_model(self):
        # from modelscope import pipeline_cls
        module = getattr(importlib.import_module('modelscope'), self.pipeline_cls)

        logger.info(f'Loading model {self.model_id} with {self.pipeline_cls} ...')

        self.model = module.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
            **self.kwargs,
        )

        self.model.to(self.device)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def get_local_model(task_cfg: 'TaskConfig') -> Optional[LocalModel]:
    """Get the base local model for the task. If the task is not checkpoint-based, return None.
       Avoids loading model multiple times for different datasets.
    """
    if task_cfg.eval_type != EvalType.CHECKPOINT:
        return None
    elif task_cfg.model_task == ModelTask.TEXT_GENERATION:
        base_model = LocalChatModel(model_id=task_cfg.model, **task_cfg.model_args)
        base_model.load_model()
        return base_model
    elif task_cfg.model_task == ModelTask.IMAGE_GENERATION:
        base_model = LocalImageModel(model_id=task_cfg.model, **task_cfg.model_args)
        base_model.load_model()
        return base_model
    else:
        raise ValueError(f'Unsupported model task: {task_cfg.model_task} for model checkpoint.')
