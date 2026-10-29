"""With stream=True, the litellm model's latency covers reading the stream, as in the openai and
anthropic models, so it is never shorter than the time to first token."""
import time
from typing import Any

import pytest
from openai.types.chat import ChatCompletion

litellm = pytest.importorskip('litellm')

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig
from evalscope.models import litellm_compatible
from evalscope.models.litellm_compatible import LiteLLMAPI

STREAM_SECONDS = 0.2


def _completion_response() -> ChatCompletion:
    return ChatCompletion.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion',
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'answer'}}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    })


def _slow_stream_collection(
    response: Any, request_start: float, finish_reason_chunks: Any = None
) -> tuple[ChatCompletion, float]:
    time.sleep(STREAM_SECONDS)  # the chunks arrive while the stream is read
    return _completion_response(), 0.01


def test_stream_latency_includes_reading_the_stream(monkeypatch: Any) -> None:
    monkeypatch.setattr(litellm, 'completion', lambda **request: iter(()))
    monkeypatch.setattr(litellm_compatible, 'collect_stream_response', _slow_stream_collection)

    api = LiteLLMAPI(model_name='openai/test-model')
    output = api.generate(
        input=[ChatMessageUser(content='question')],
        tools=[],
        tool_choice='none',
        config=GenerateConfig(stream=True),
    )

    assert output.time >= STREAM_SECONDS
