"""Optional registration for the text-only architecture in the PPU vendor build."""


def register() -> None:
    """Register an existing implementation without replacing vLLM or its PPU kernels."""
    from vllm import ModelRegistry

    if 'Qwen3_5ForCausalLM' not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            'Qwen3_5ForCausalLM',
            'decision_ppu_model:PPUQwen3_5ForCausalLM',
        )
