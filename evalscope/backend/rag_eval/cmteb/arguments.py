from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any


@dataclass
class ModelArguments:
    # Arguments for embeding model: sentence transformer or cross encoder
    model_name_or_path: str = ""  # model name or path
    is_cross_encoder: bool = False  # whether the model is a cross encoder
    # pooling mode: Either “cls”, “lasttoken”, “max”, “mean”, “mean_sqrt_len_tokens”, or “weightedmean”.
    pooling_mode: Optional[str] = None
    max_seq_length: int = 512  # max sequence length
    # prompt for llm based model
    prompt: str = ""
    # model kwargs
    model_kwargs: dict = field(default_factory=lambda: {"torch_dtype": "auto"})
    # config kwargs
    config_kwargs: Dict[str, Any] = field(default_factory=dict)
    # encode kwargs
    encode_kwargs: dict = field(
        default_factory=lambda: {"show_progress_bar": True, "batch_size": 32}
    )
    hub: str = "modelscope"  # modelscope or huggingface

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "is_cross_encoder": self.is_cross_encoder,
            "pooling_mode": self.pooling_mode,
            "max_seq_length": self.max_seq_length,
            "prompt": self.prompt,
            "model_kwargs": self.model_kwargs,
            "config_kwargs": self.config_kwargs,
            "encode_kwargs": self.encode_kwargs,
            "hub": self.hub,
        }


@dataclass
class EvalArguments:
    # Evaluation
    tasks: List[str] = field(default_factory=list)  # task names
    verbosity: int = 2  # verbosity level 0-3
    output_folder: str = "outputs"  # output folder
    overwrite_results: bool = True  # overwrite results
    limits: Optional[int] = None  # limit number of samples
    hub: str = "modelscope"  # modelscope or huggingface
    top_k: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks": self.tasks,
            "verbosity": self.verbosity,
            "output_folder": self.output_folder,
            "overwrite_results": self.overwrite_results,
            "limits": self.limits,
            "hub": self.hub,
            "top_k": 5,
        }
