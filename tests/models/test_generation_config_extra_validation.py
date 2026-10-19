from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, Model, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo


class _TextAPI(ModelAPI):

    def generate(self, input: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice,
                 config: GenerateConfig) -> ModelOutput:
        return ModelOutput.from_content(model=self.model_name, content='ok')


def test_model_preserves_extra_generation_config_fields() -> None:
    class _ExtraAPI(_TextAPI):

        def generate(self, input: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice,
                     config: GenerateConfig) -> ModelOutput:
            assert config.model_extra == {'base_extra': 'base', 'call_extra': 'call'}
            return ModelOutput.from_content(model=self.model_name, content='ok')

    model = Model(_ExtraAPI('test'), GenerateConfig(base_extra='base'))

    assert model.generate('hello', config=GenerateConfig(call_extra='call')).message.content == 'ok'


def test_text_model_allows_provider_extra_body() -> None:
    model = Model(_TextAPI('test'), GenerateConfig(extra_body={'enable_thinking': True}))

    assert model.generate('hello').message.content == 'ok'


def test_text_model_keeps_extra_generation_config_fields() -> None:
    model = Model(_TextAPI('test'), GenerateConfig(custom_provider_option='fast'))

    assert model.generate('hello').message.content == 'ok'


def test_merge_keeps_user_settings_the_per_call_config_did_not_set() -> None:
    user = GenerateConfig(retries=1, retry_interval=0, temperature=0.2)
    per_call = GenerateConfig(max_tokens=64)

    merged = user.merge(per_call)

    assert (merged.retries, merged.retry_interval, merged.temperature) == (1, 0, 0.2)
    assert merged.max_tokens == 64
    assert user.merge(GenerateConfig(retries=3)).retries == 3
