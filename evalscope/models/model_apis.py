from evalscope.api.model import ModelAPI
from evalscope.api.registry import register_model_api
from evalscope.utils.deprecation_utils import deprecated


@register_model_api(name='mock_llm')
def mockllm() -> type[ModelAPI]:
    from .mockllm import MockLLM

    return MockLLM


@register_model_api(name='openai_api')
def openai_api() -> type[ModelAPI]:
    from .openai_compatible import OpenAICompatibleAPI

    return OpenAICompatibleAPI


@register_model_api(name='server')
@deprecated(since='1.0.0', remove_in='1.1.0', alternative='openai_api')
def server() -> type[ModelAPI]:
    from .openai_compatible import OpenAICompatibleAPI

    return OpenAICompatibleAPI


@register_model_api(name='llm_ckpt')
def llm_ckpt() -> type[ModelAPI]:
    from .modelscope import ModelScopeAPI

    return ModelScopeAPI


@register_model_api(name='checkpoint')
@deprecated(since='1.0.0', remove_in='1.1.0', alternative='llm_ckpt')
def checkpoint() -> type[ModelAPI]:
    from .modelscope import ModelScopeAPI

    return ModelScopeAPI


@register_model_api(name='text2image')
def text2image() -> type[ModelAPI]:
    from .text2image_model import Text2ImageAPI

    return Text2ImageAPI
