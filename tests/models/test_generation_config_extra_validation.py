import pytest

from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, Model, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo


class _TextAPI(ModelAPI):

    def generate(self, input: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice,
                 config: GenerateConfig) -> ModelOutput:
        return ModelOutput.from_content(model=self.model_name, content='ok')


class _ImageAPI(_TextAPI):
    allows_generation_config_extras = True


def test_text_model_rejects_unknown_generation_config_fields() -> None:
    with pytest.raises(ValueError, match='override_cpus'):
        Model(_TextAPI('test'), GenerateConfig(override_cpus=32))


def test_text_model_rejects_unknown_per_call_generation_config_fields() -> None:
    model = Model(_TextAPI('test'), GenerateConfig())

    with pytest.raises(ValueError, match='override_cpus'):
        model.generate('hello', config=GenerateConfig(override_cpus=32))


def test_text_model_allows_provider_extra_body() -> None:
    model = Model(_TextAPI('test'), GenerateConfig(extra_body={'enable_thinking': True}))

    assert model.generate('hello').message.content == 'ok'


def test_image_model_keeps_generation_config_extras() -> None:
    model = Model(_ImageAPI('test'), GenerateConfig(custom_scheduler='fast'))

    assert model.generate('hello').message.content == 'ok'
