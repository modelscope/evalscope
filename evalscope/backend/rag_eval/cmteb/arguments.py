from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import torch


@dataclass
class Arguments:
    # Model
    model_name_or_path: str = ""  # model name or path
    is_cross_encoder: bool = False  # whether the model is a cross encoder
    # pooling mode: Either “cls”, “lasttoken”, “max”, “mean”, “mean_sqrt_len_tokens”, or “weightedmean”.
    pooling_mode: str = "cls"
    max_seq_length: int = 512  # max sequence length
    # model kwargs
    model_kwargs: dict = field(default_factory=lambda: {"torch_dtype": torch.bfloat16})
    # config kwargs
    config_kwargs: Dict[str, Any] = field(default_factory=dict)
    # encode kwargs
    encode_kwargs: dict = field(
        default_factory=lambda: {
            "show_progress_bar": True,
            "batch_size": 32,
            "normalize_embeddings": True,
        }
    )

    # Evaluation
    tasks: List[str] = field(default_factory=list)  # task names
    # instructions with task name as key
    instructions: Dict[str, Any] = field(default_factory=dict)
    verbosity: int = 2  # verbosity level 0-3
    output_folder: str = "outputs"  # output folder
    overwrite_results: bool = True  # overwrite results
    limits: Optional[int] = None  # limit number of samples
    hub: str = "modelscope"  # modelscope or huggingface
