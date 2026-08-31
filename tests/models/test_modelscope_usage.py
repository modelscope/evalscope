from typing import Any, List

import pytest

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig
from evalscope.models import modelscope as modelscope_module
from evalscope.models.modelscope import GenerateOutput, ModelScopeAPI


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


def test_generate_aggregates_usage_across_choices(monkeypatch: Any) -> None:
    responses: List[GenerateOutput] = [
        GenerateOutput(
            output='first',
            input_tokens=10,
            output_tokens=4,
            total_tokens=14,
            logprobs=None,
            time=1.2,
            stop_reason='stop',
        ),
        GenerateOutput(
            output='second',
            input_tokens=10,
            output_tokens=6,
            total_tokens=16,
            logprobs=None,
            time=1.5,
            stop_reason='max_tokens',
        ),
    ]
    monkeypatch.setattr(modelscope_module, 'batched_generate', lambda _input: responses)

    output = _make_api().generate(
        input=[ChatMessageUser(content='hi')],
        tools=[],
        tool_choice='none',
        config=GenerateConfig(),
    )

    assert [choice.message.text for choice in output.choices] == ['first', 'second']
    # usage must aggregate over all returned choices, not just the last one
    assert output.usage.input_tokens == 20
    assert output.usage.output_tokens == 10
    assert output.usage.total_tokens == 30
    assert output.time == pytest.approx(1.5)
    assert output.message.perf_metrics.input_tokens == 20
    assert output.message.perf_metrics.output_tokens == 10
