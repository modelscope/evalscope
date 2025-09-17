from evalscope.api.model import ModelAPI
from evalscope.api.registry import register_model_api
from evalscope.utils.deprecation_utils import deprecated
from evalscope.utils.import_utils import check_import


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
    check_import('torch', package='torch', raise_error=True, feature_name='llm_ckpt')

    from .modelscope import ModelScopeAPI

    return ModelScopeAPI


@register_model_api(name='checkpoint')
@deprecated(since='1.0.0', remove_in='1.1.0', alternative='llm_ckpt')
def checkpoint() -> type[ModelAPI]:
    check_import('torch', package='torch', raise_error=True, feature_name='llm_ckpt')

    from .modelscope import ModelScopeAPI

    return ModelScopeAPI


@register_model_api(name='text2image')
def text2image() -> type[ModelAPI]:
    check_import(['torch', 'torchvision', 'diffusers'],
                 package='evalscope[aigc]',
                 raise_error=True,
                 feature_name='text2image')

    from .text2image_model import Text2ImageAPI

    return Text2ImageAPI


@register_model_api(name='image_editing')
def image_editing() -> type[ModelAPI]:
    check_import(['torch', 'torchvision', 'diffusers'],
                 package='evalscope[aigc]',
                 raise_error=True,
                 feature_name='image_editing')

    from .image_edit_model import ImageEditAPI

    return ImageEditAPI
