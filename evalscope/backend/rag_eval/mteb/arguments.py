# Copyright (c) Alibaba, Inc. and its affiliates.
from pydantic import Field, field_validator
from typing import Any, Dict, List, Literal, Optional

from evalscope.utils.argument_utils import BaseArgument


class MTEBModelConfig(BaseArgument):
    """MTEB model configuration."""

    model_name_or_path: str
    is_cross_encoder: bool = False
    pooling_mode: Optional[str] = None
    max_seq_length: int = 512
    prompt: Optional[str] = None
    prompts: Optional[Dict[str, str]] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = Field(default_factory=lambda: {'batch_size': 32})
    hub: str = 'modelscope'
    # API model fields
    model_name: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    dimensions: Optional[int] = None


class MTEBEvalConfig(BaseArgument):
    """MTEB evaluation configuration."""

    task_names: Optional[List[str]] = None
    task_types: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    custom_tasks: Optional[List[Dict[str, Any]]] = None
    output_folder: str = 'outputs'
    overwrite_results: bool = True
    limits: Optional[int] = None
    hub: str = 'modelscope'
    top_k: int = 10
    splits: Optional[Dict[str, Any]] = None
    encode_kwargs: Optional[Dict[str, Any]] = None


class MTEBToolConfig(BaseArgument):
    """Complete configuration for tool='mteb' in eval_config."""

    tool: Literal['mteb'] = 'mteb'
    models: List[MTEBModelConfig]
    eval: MTEBEvalConfig

    @field_validator('tool', mode='before')
    @classmethod
    def normalize_tool(cls, v):
        return v.lower() if isinstance(v, str) else v

    @field_validator('models', mode='before')
    @classmethod
    def parse_models(cls, v):
        if isinstance(v, list):
            return [MTEBModelConfig(**m) if isinstance(m, dict) else m for m in v]
        return v

    @field_validator('eval', mode='before')
    @classmethod
    def parse_eval(cls, v):
        if isinstance(v, dict):
            return MTEBEvalConfig(**v)
        return v
