import torch
from typing import TYPE_CHECKING, Optional

from evalscope.constants import DEFAULT_MODEL_CACHE_DIR, DEFAULT_MODEL_REVISION, EvalType
from evalscope.utils.logger import get_logger
from evalscope.utils.model_utils import get_device

if TYPE_CHECKING:
    from evalscope.config import TaskConfig

logger = get_logger()


class LocalModel:

    def __init__(self,
                 model_id: str,
                 model_revision: str = DEFAULT_MODEL_REVISION,
                 device_map: str = 'auto',
                 torch_dtype: str = 'auto',
                 cache_dir: str = None,
                 **kwargs):
        from modelscope import AutoModelForCausalLM, AutoTokenizer

        model_cache_dir = cache_dir or DEFAULT_MODEL_CACHE_DIR

        if isinstance(torch_dtype, str) and torch_dtype != 'auto':
            torch_dtype = eval(torch_dtype)

        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device_map

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=model_revision,
            trust_remote_code=True,
            cache_dir=model_cache_dir,
        )

        # Fix no padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=model_revision,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            cache_dir=model_cache_dir,
        )

        self.model_cfg = {
            'model_id': model_id,
            'device_map': device_map,
            'torch_dtype': str(torch_dtype),
        }


def get_local_model(task_cfg: 'TaskConfig') -> Optional[LocalModel]:
    """Get the base local model for the task. If the task is not checkpoint-based, return None.
       Avoids loading model multiple times for different datasets.
    """
    if task_cfg.eval_type != EvalType.CHECKPOINT:
        return None
    else:
        device_map = task_cfg.model_args.get('device_map', get_device())
        cache_dir = task_cfg.model_args.get('cache_dir', None)
        model_precision = task_cfg.model_args.get('precision', 'torch.float16')
        model_revision = task_cfg.model_args.get('revision', DEFAULT_MODEL_REVISION)

        base_model = LocalModel(
            model_id=task_cfg.model,
            model_revision=model_revision,
            device_map=device_map,
            torch_dtype=model_precision,
            cache_dir=cache_dir)
        return base_model
