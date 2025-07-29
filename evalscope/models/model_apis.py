from evalscope.api.model import ModelAPI
from evalscope.api.registry import register_model_api


@register_model_api(name='mockllm')
def mockllm() -> type[ModelAPI]:
    from .mockllm import MockLLM

    return MockLLM
