"""Generate-level regression for local tool calls and their stop reason.

``ModelScopeAPI.generate`` reads tool calls out of the checkpoint's text, but the
local generation path reports ``stop``/``max_tokens`` as its stop reason. A
benchmark that keys on ``stop_reason == 'tool_calls'`` (as the hosted paths
report) would then miss a call the parser did find, so the generate path has to
report ``tool_calls`` whenever it parsed one and leave the original reason alone
for ordinary text.
"""
from typing import Any, List

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig
from evalscope.models import modelscope as modelscope_module
from evalscope.models.modelscope import GenerateOutput, ModelScopeAPI

CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>'


class _FakeTokenizer:

    chat_template = None

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        raise AssertionError('tokenizer must not be called when batched_generate is mocked')

    def batch_decode(self, *args: Any, **kwargs: Any) -> list:
        raise AssertionError('batch_decode must not be called when batched_generate is mocked')


class _FakeModel:

    device = 'cpu'

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('model.generate must not be called when batched_generate is mocked')


def _make_api() -> ModelScopeAPI:
    """Build a ModelScopeAPI without loading any weights."""
    api = object.__new__(ModelScopeAPI)
    api.model_name = 'test-model'
    api.tokenizer = _FakeTokenizer()
    api.model = _FakeModel()
    api.chat_template = None
    api.tokenizer_call_args = {}
    api.enable_thinking = None
    return api


def _generate(monkeypatch: Any, output: str, stop_reason: str) -> Any:
    responses: List[GenerateOutput] = [
        GenerateOutput(
            output=output,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            logprobs=None,
            time=0.1,
            stop_reason=stop_reason,
        )
    ]
    monkeypatch.setattr(modelscope_module, 'batched_generate', lambda _input: responses)
    return _make_api().generate(
        input=[ChatMessageUser(content='hi')],
        tools=[],
        tool_choice='none',
        config=GenerateConfig(),
    )


def test_generate_reports_tool_calls_stop_reason(monkeypatch: Any) -> None:
    output = _generate(monkeypatch, f'Let me check.\n{CALL}', 'stop')

    choice = output.choices[0]
    assert choice.stop_reason == 'tool_calls'
    assert choice.message.text == 'Let me check.'
    assert choice.message.tool_calls is not None and len(choice.message.tool_calls) == 1
    assert choice.message.tool_calls[0].function.name == 'get_weather'


def test_generate_preserves_stop_reason_without_tool_call(monkeypatch: Any) -> None:
    output = _generate(monkeypatch, 'plain answer', 'max_tokens')

    choice = output.choices[0]
    assert choice.stop_reason == 'max_tokens'
    assert choice.message.tool_calls is None
    assert choice.message.text == 'plain answer'
