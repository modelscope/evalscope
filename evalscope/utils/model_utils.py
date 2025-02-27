import os
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple, Union

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
