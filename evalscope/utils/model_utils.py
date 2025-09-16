import numpy as np
import random
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from evalscope.utils.import_utils import check_import

if TYPE_CHECKING:
    from transformers import GenerationConfig


class EvalBackend(Enum):
    #  NOTE: compatible with ms-swfit v2.x
    NATIVE = 'Native'
    OPEN_COMPASS = 'OpenCompass'
    VLM_EVAL_KIT = 'VLMEvalKit'
    RAG_EVAL = 'RAGEval'
    THIRD_PARTY = 'ThirdParty'


def fix_do_sample_warning(generation_config: 'GenerationConfig') -> None:
    # Use the default values of temperature/top_p/top_k in generation_config.
    if generation_config.temperature == 0:
        generation_config.do_sample = False
    if generation_config.do_sample is False:
        generation_config.temperature = 1.
        generation_config.top_p = 1.
        generation_config.top_k = 50


def get_device() -> str:
    from transformers.utils import is_torch_cuda_available, is_torch_mps_available, is_torch_npu_available

    if is_torch_npu_available():
        device = 'npu'
    elif is_torch_mps_available():
        device = 'mps'
    elif is_torch_cuda_available():
        device = 'cuda'
    else:
        device = 'cpu'

    return device


def dict_torch_dtype_to_str(d: Dict[str, Any]) -> dict:
    """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.

        Refer to: https://github.com/huggingface/transformers/pull/16065/files for details.
        """
    if d.get('torch_dtype', None) is not None and not isinstance(d['torch_dtype'], str):
        d['torch_dtype'] = str(d['torch_dtype']).split('.')[1]

    for value in d.values():
        if isinstance(value, dict):
            dict_torch_dtype_to_str(value)

    return d


def seed_everything(seed: int):
    """Set all random seeds to a fixed value for reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    if check_import('torch', raise_warning=False):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
