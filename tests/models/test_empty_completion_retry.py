"""Tests for gateway errors deserialized as empty chat completions."""

import asyncio
import json
from types import SimpleNamespace
from typing import Optional

import pytest
from openai.types.chat import ChatCompletion

from evalscope.api.model import GenerateConfig
from evalscope.models.openai_compatible import EmptyCompletionError, OpenAICompatibleAPI

# Captured verbatim from an OpenRouter free-tier request that failed 13s after
# the keepalive had already committed 200 OK: whitespace padding, then the error.
GATEWAY_ERROR_BODY = ('\n         \n' * 31) + '{"error":{"message":"Insufficient balance","code":402}}'


def _error_payload_completion(
    code: int = 402, message: str = 'Insufficient balance'
) -> ChatCompletion:
    """Deserialize an error body the way the OpenAI SDK does for a 200 response."""
    payload = json.loads(GATEWAY_ERROR_BODY)
    payload['error'] = {'message': message, 'code': code}
    return ChatCompletion.construct(**payload)


def _valid_completion(content: str = 'complete response') -> ChatCompletion:
    return ChatCompletion.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion',
        'choices': [{
            'index': 0,
            'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': content},
        }],
    })


def _prepare_api(monkeypatch: pytest.MonkeyPatch) -> OpenAICompatibleAPI:
    api = object.__new__(OpenAICompatibleAPI)
    api.base_url = 'https://example.test/v1'
    api.model_name = 'test-model'
    api.resolve_tools = lambda tools, tool_choice, config: (tools, tool_choice, config)
    api.completion_params = lambda config, tools: {'model': 'test-model'}
    api.validate_request_params = lambda request: None
    api.on_response = lambda response: None
    api.chat_choices_from_completion = lambda completion, tools: []

    monkeypatch.setattr('evalscope.models.openai_compatible.openai_chat_messages', lambda *args, **kwargs: [])

    def model_output(completion, choices):
        return SimpleNamespace(
            content=completion.choices[0].message.content,
            usage=None,
            message=SimpleNamespace(),
            time=None,
        )

    monkeypatch.setattr('evalscope.models.openai_compatible.model_output_from_openai', model_output)
    return api


def test_sdk_deserializes_error_payload_as_a_choiceless_completion() -> None:
    """The premise: an error body becomes a ChatCompletion that looks like success."""
    completion = _error_payload_completion()

    assert completion.choices is None
    assert getattr(completion, 'id', None) is None
    # The gateway's real error survives here, which is what the retry reports.
    assert completion.model_extra == {'error': {'message': 'Insufficient balance', 'code': 402}}


@pytest.mark.parametrize('code', [400, 401, 402, 403, 404, 422])
def test_generate_does_not_retry_non_retryable_gateway_errors(monkeypatch, code) -> None:
    api = _prepare_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        return _error_payload_completion(code=code, message='client error')

    api.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with pytest.raises(ValueError, match='client error'):
        api.generate([], [], None, GenerateConfig(retries=5, retry_interval=0))

    assert attempts == 1


def test_generate_async_does_not_retry_non_retryable_gateway_error(monkeypatch) -> None:
    api = _prepare_api(monkeypatch)
    attempts = 0

    async def create(**request):
        nonlocal attempts
        attempts += 1
        return _error_payload_completion()

    async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(OpenAICompatibleAPI, 'async_client', property(lambda self: async_client))

    with pytest.raises(ValueError, match='Insufficient balance'):
        asyncio.run(api.generate_async([], [], None, GenerateConfig(retries=5, retry_interval=0)))

    assert attempts == 1


def test_generate_retries_when_gateway_returns_error_payload(monkeypatch) -> None:
    api = _prepare_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            return _error_payload_completion(code=503, message='Provider unavailable')
        return _valid_completion()

    api.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    result = api.generate([], [], None, GenerateConfig(retries=5, retry_interval=0))

    assert attempts == 3
    assert result.content == 'complete response'


def test_generate_async_retries_when_gateway_returns_error_payload(monkeypatch) -> None:
    api = _prepare_api(monkeypatch)
    attempts = 0

    async def create(**request):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            return _error_payload_completion(code=503, message='Provider unavailable')
        return _valid_completion()

    async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(OpenAICompatibleAPI, 'async_client', property(lambda self: async_client))

    result = asyncio.run(api.generate_async([], [], None, GenerateConfig(retries=5, retry_interval=0)))

    assert attempts == 3
    assert result.content == 'complete response'


def test_exhausted_retries_report_the_gateway_error(monkeypatch) -> None:
    api = _prepare_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        return _error_payload_completion(code=503, message='Provider unavailable')

    api.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with pytest.raises(EmptyCompletionError, match='Provider unavailable') as exc_info:
        api.generate([], [], None, GenerateConfig(retries=3, retry_interval=0))

    assert isinstance(exc_info.value, ValueError)
    assert attempts == 3


def test_choiceless_completion_without_error_detail_still_retries(monkeypatch) -> None:
    api = _prepare_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        return ChatCompletion.construct()

    api.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with pytest.raises(EmptyCompletionError, match='no error detail'):
        api.generate([], [], None, GenerateConfig(retries=2, retry_interval=0))

    assert attempts == 2


def test_valid_completion_is_not_retried(monkeypatch) -> None:
    api = _prepare_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        return _valid_completion('first response')

    api.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    result = api.generate([], [], None, GenerateConfig(retries=5, retry_interval=0))

    assert attempts == 1
    assert result.content == 'first response'
