import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from torch import dtype

from evalscope.constants import DEFAULT_MODEL_CACHE_DIR
from evalscope.utils.logger import get_logger

logger = get_logger()


class LocalModel:

    def __init__(self,
                 model_id: str,
                 model_revision: str = 'master',
                 device_map: str = 'auto',
                 torch_dtype: dtype = torch.bfloat16,
                 cache_dir: str = None,
                 **kwargs):
        model_cache_dir = cache_dir or DEFAULT_MODEL_CACHE_DIR

        self.model_id = model_id
        self.model_revision = model_revision
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Device: {self.device}')

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=model_revision,
            trust_remote_code=True,
            cache_dir=model_cache_dir,
        )

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
