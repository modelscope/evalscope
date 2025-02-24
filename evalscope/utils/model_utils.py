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


def get_dist_setting() -> Tuple[int, int, int, int]:
    """return rank, local_rank, world_size, local_world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    # compat deepspeed launch
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', None) or os.getenv('LOCAL_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def get_device(rank: Optional[Union[str, int]] = None) -> str:
    from transformers.utils import is_torch_cuda_available, is_torch_mps_available, is_torch_npu_available
    if rank is None:
        rank = get_dist_setting()[1]
        if rank < 0 or rank is None:
            rank = 0
    if isinstance(rank, int):
        rank = str(rank)
    if is_torch_npu_available():
        device = 'npu:{}'.format(rank)
    elif is_torch_mps_available():
        device = 'mps:{}'.format(rank)
    elif is_torch_cuda_available():
        device = 'cuda:{}'.format(rank)
    else:
        device = 'cpu'

    return device
