"""Reuse the vendor's text model and its existing hybrid-cache implementation."""

import torch
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration


class PPUQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    """Expose the hybrid-cache capabilities required by the native GDN layers."""

    is_hybrid = True
    supports_mrope = True
    embedding_modules = {'embed_tokens': 'input_embeddings', 'lm_head': 'output_embeddings'}
    get_mamba_state_dtype_from_config = Qwen3_5ForConditionalGeneration.__dict__['get_mamba_state_dtype_from_config']
    get_mamba_state_shape_from_config = Qwen3_5ForConditionalGeneration.__dict__['get_mamba_state_shape_from_config']
    get_mamba_state_copy_func = Qwen3_5ForConditionalGeneration.__dict__['get_mamba_state_copy_func']

    def get_mrope_input_positions(self, input_tokens: list[int], mm_features: list) -> tuple[torch.Tensor, int]:
        """Text has identical temporal/height/width positions and zero position delta."""
        if mm_features:
            raise ValueError('This compatibility adapter supports text-only decisions.')
        return torch.arange(len(input_tokens), dtype=torch.int64).expand(3, -1), 0
