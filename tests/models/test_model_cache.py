from typing import List

import pytest

from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput, get_model
from evalscope.api.model.model import ModelCache
from evalscope.api.registry import MODEL_APIS
from evalscope.api.tool import ToolChoice, ToolInfo


class FakeOpenAIBackend(ModelAPI):

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        return ModelOutput(model=self.model_name, choices=[])


class FakeLiteLLMBackend(ModelAPI):

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        return ModelOutput(model=self.model_name, choices=[])


@pytest.fixture
def fake_backends():
    MODEL_APIS['fake_openai_backend'] = FakeOpenAIBackend
    MODEL_APIS['fake_litellm_backend'] = FakeLiteLLMBackend
    ModelCache._models.clear()
    yield
    MODEL_APIS.pop('fake_openai_backend', None)
    MODEL_APIS.pop('fake_litellm_backend', None)
    ModelCache._models.clear()


def test_model_cache_key_includes_eval_type(fake_backends) -> None:
    first = get_model(model='shared-model', eval_type='fake_openai_backend', api_key='key')
    second = get_model(model='shared-model', eval_type='fake_litellm_backend', api_key='key')

    assert first is not second
    assert isinstance(first.api, FakeOpenAIBackend)
    assert isinstance(second.api, FakeLiteLLMBackend)


def test_same_eval_type_returns_memoized_model(fake_backends) -> None:
    first = get_model(model='shared-model', eval_type='fake_openai_backend', api_key='key')
    again = get_model(model='shared-model', eval_type='fake_openai_backend', api_key='key')

    assert again is first
