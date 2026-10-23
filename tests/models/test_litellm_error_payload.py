"""A 200 response carrying an error payload (no choices) must fail, not silently succeed.

LiteLLM routes to 100+ providers; some return HTTP 200 with an error body. The
non-streaming path now checks the raw response via ``raise_if_error_payload``
(before re-validation), and the streaming path relies on the in-band error
detection added to ``collect_stream_response``. See modelscope/evalscope#1909.
"""

from typing import Any, List

import litellm
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig
from evalscope.models.litellm_compatible import LiteLLMAPI
from evalscope.models.utils.openai import EmptyCompletionError


def _error_completion(code: int = 503, message: str = 'Insufficient balance') -> ChatCompletion:
    """A 200 body the SDK deserialized (via construct) with choices=None."""
    return ChatCompletion.construct(
        id=None,
        created=None,
        model=None,
        object='chat.completion',
        choices=None,
        error={'message': message, 'code': code},
    )


def _chunk(content: str) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion.chunk',
        'choices': [{'index': 0, 'finish_reason': None, 'delta': {'content': content}}],
    })


def _error_chunk() -> ChatCompletionChunk:
    return ChatCompletionChunk.construct(
        id=None, created=None, model=None, object='error', choices=None,
        message='internal error', type='internal_error',
    )


def _input() -> List[Any]:
    return [ChatMessageUser(content='question')]


def test_non_stream_error_payload_raises(monkeypatch: Any) -> None:
    monkeypatch.setattr(litellm, 'completion', lambda **request: _error_completion())

    api = LiteLLMAPI(model_name='openai/test-model')

    with pytest.raises(EmptyCompletionError, match='no choices'):
        api.generate(input=_input(), tools=[], tool_choice='none', config=GenerateConfig())


def test_non_stream_non_retryable_error_does_not_retry(monkeypatch: Any) -> None:
    attempts = 0

    def _completion(**request: Any) -> ChatCompletion:
        nonlocal attempts
        attempts += 1
        return _error_completion(code=400, message='bad request')

    monkeypatch.setattr(litellm, 'completion', _completion)
    api = LiteLLMAPI(model_name='openai/test-model')

    with pytest.raises(EmptyCompletionError):
        api.generate(input=_input(), tools=[], tool_choice='none', config=GenerateConfig(retries=5, retry_interval=0))

    # A non-retryable gateway code maps to _NonRetryableEmptyCompletionError,
    # which is in NON_RETRYABLE_LITELLM_ERRORS, so the single attempt stands.
    assert attempts == 1


def test_stream_error_event_raises(monkeypatch: Any) -> None:
    monkeypatch.setattr(litellm, 'completion', lambda **request: [_chunk('partial'), _error_chunk()])

    api = LiteLLMAPI(model_name='openai/test-model')

    with pytest.raises(EmptyCompletionError, match='error event'):
        api.generate(
            input=_input(), tools=[], tool_choice='none', config=GenerateConfig(stream=True)
        )
