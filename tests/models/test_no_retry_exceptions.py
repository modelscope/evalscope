"""Non-retryable 4xx client errors must escape retry_call immediately.

Each model backend degrades client errors (context length exceeded, invalid
parameters, bad credentials) in its own exception handler instead of failing the
run, so retrying them inside retry_call only burns ``retries * retry_interval``
per failing sample. These tests exercise the backend call sites with mock SDK
clients; no network access is performed.
"""

import asyncio
from types import SimpleNamespace
from typing import Type

import anthropic
import httpx
import openai
import pytest

from evalscope.api.model import GenerateConfig
from evalscope.models.anthropic_compatible import AnthropicCompatibleAPI
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from evalscope.models.openai_responses import OpenAIResponsesAPI

OPENAI_CLIENT_ERROR_CASES = (
    pytest.param(openai.BadRequestError, 400, id='bad-request'),
    pytest.param(openai.AuthenticationError, 401, id='authentication'),
    pytest.param(openai.PermissionDeniedError, 403, id='permission-denied'),
    pytest.param(openai.NotFoundError, 404, id='not-found'),
    pytest.param(openai.UnprocessableEntityError, 422, id='unprocessable-entity'),
)

ANTHROPIC_CLIENT_ERROR_CASES = (
    pytest.param(anthropic.BadRequestError, 400, id='bad-request'),
    pytest.param(anthropic.AuthenticationError, 401, id='authentication'),
    pytest.param(anthropic.PermissionDeniedError, 403, id='permission-denied'),
    pytest.param(anthropic.NotFoundError, 404, id='not-found'),
    pytest.param(anthropic.UnprocessableEntityError, 422, id='unprocessable-entity'),
)


def _openai_client_error(
    error_type: Type[openai.APIStatusError],
    status_code: int,
    message: str = 'non-retryable client error',
) -> openai.APIStatusError:
    response = httpx.Response(
        status_code,
        request=httpx.Request('POST', 'https://example.test/v1/chat/completions'),
        json={'error': {'message': message}},
    )
    return error_type(
        f'Error code: {status_code} - {message}',
        response=response,
        body=response.json(),
    )


def _anthropic_client_error(
    error_type: Type[anthropic.APIStatusError],
    status_code: int,
    message: str = 'non-retryable client error',
) -> anthropic.APIStatusError:
    response = httpx.Response(
        status_code,
        request=httpx.Request('POST', 'https://example.test/v1/messages'),
        json={'type': 'error', 'error': {'type': 'invalid_request_error', 'message': message}},
    )
    return error_type(f'Error code: {status_code} - {message}', response=response, body=response.json())


def _prepare_openai_api(monkeypatch: pytest.MonkeyPatch) -> OpenAICompatibleAPI:
    api = object.__new__(OpenAICompatibleAPI)
    api.base_url = 'https://example.test/v1'
    api.model_name = 'test-model'
    api.resolve_tools = lambda tools, tool_choice, config: (tools, tool_choice, config)
    api.completion_params = lambda config, tools: {'model': 'test-model'}
    api.validate_request_params = lambda request: None

    monkeypatch.setattr('evalscope.models.openai_compatible.openai_chat_messages', lambda *args, **kwargs: [])
    return api


def _prepare_anthropic_api(monkeypatch: pytest.MonkeyPatch) -> AnthropicCompatibleAPI:
    api = object.__new__(AnthropicCompatibleAPI)
    api.model_name = 'test-model'
    api.resolve_tools = lambda tools, tool_choice, config: (tools, tool_choice, config)
    api.completion_params = lambda config: {'model': 'test-model'}
    api.explicit_cache_control_params = lambda config: None
    api.validate_request_params = lambda request: None

    monkeypatch.setattr(
        'evalscope.models.anthropic_compatible.anthropic_chat_messages', lambda *args, **kwargs: (None, [])
    )
    return api


def _prepare_openai_responses_api() -> OpenAIResponsesAPI:
    api = object.__new__(OpenAIResponsesAPI)
    api.model_name = 'test-model'
    api._build_request = lambda input, tools, tool_choice, config: ({}, tools, config)
    return api


@pytest.mark.parametrize(('error_type', 'status_code'), OPENAI_CLIENT_ERROR_CASES)
def test_openai_generate_does_not_retry_client_errors(monkeypatch, error_type, status_code) -> None:
    api = _prepare_openai_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        raise _openai_client_error(error_type, status_code)

    api.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with pytest.raises(error_type):
        api.generate([], [], None, GenerateConfig(retries=5, retry_interval=0))

    assert attempts == 1


@pytest.mark.parametrize(('error_type', 'status_code'), OPENAI_CLIENT_ERROR_CASES)
def test_openai_generate_async_does_not_retry_client_errors(monkeypatch, error_type, status_code) -> None:
    api = _prepare_openai_api(monkeypatch)
    attempts = 0

    async def create(**request):
        nonlocal attempts
        attempts += 1
        raise _openai_client_error(error_type, status_code)

    async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(OpenAICompatibleAPI, 'async_client', property(lambda self: async_client))

    with pytest.raises(error_type):
        asyncio.run(api.generate_async([], [], None, GenerateConfig(retries=5, retry_interval=0)))

    assert attempts == 1


def test_openai_bad_request_still_degrades_gracefully(monkeypatch) -> None:
    api = _prepare_openai_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        raise _openai_client_error(openai.BadRequestError, 400, 'maximum context length is 4096 tokens')

    api.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    result = api.generate([], [], None, GenerateConfig(retries=5, retry_interval=0))

    assert attempts == 1
    assert '400' in result.choices[0].message.content


def test_openai_generate_still_retries_connection_errors(monkeypatch) -> None:
    api = _prepare_openai_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        raise ConnectionError('stream interrupted by upstream gateway')

    api.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with pytest.raises(ConnectionError):
        api.generate([], [], None, GenerateConfig(retries=3, retry_interval=0))

    assert attempts == 3


@pytest.mark.parametrize(('error_type', 'status_code'), ANTHROPIC_CLIENT_ERROR_CASES)
def test_anthropic_generate_does_not_retry_client_errors(monkeypatch, error_type, status_code) -> None:
    api = _prepare_anthropic_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        raise _anthropic_client_error(error_type, status_code)

    api.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    with pytest.raises(error_type):
        api.generate([], [], None, GenerateConfig(retries=5, retry_interval=0))

    assert attempts == 1


@pytest.mark.parametrize(('error_type', 'status_code'), ANTHROPIC_CLIENT_ERROR_CASES)
def test_anthropic_generate_async_does_not_retry_client_errors(monkeypatch, error_type, status_code) -> None:
    api = _prepare_anthropic_api(monkeypatch)
    attempts = 0

    async def create(**request):
        nonlocal attempts
        attempts += 1
        raise _anthropic_client_error(error_type, status_code)

    async_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(AnthropicCompatibleAPI, 'async_client', property(lambda self: async_client))

    with pytest.raises(error_type):
        asyncio.run(api.generate_async([], [], None, GenerateConfig(retries=5, retry_interval=0)))

    assert attempts == 1


def test_anthropic_bad_request_still_degrades_gracefully(monkeypatch) -> None:
    api = _prepare_anthropic_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        raise _anthropic_client_error(anthropic.BadRequestError, 400, 'prompt is too long')

    api.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = api.generate([], [], None, GenerateConfig(retries=5, retry_interval=0))

    assert attempts == 1
    assert 'prompt is too long' in result.choices[0].message.content


def test_openai_responses_generate_does_not_retry_bad_request() -> None:
    api = _prepare_openai_responses_api()
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        raise _openai_client_error(openai.BadRequestError, 400)

    api.client = SimpleNamespace(responses=SimpleNamespace(create=create))

    with pytest.raises(openai.BadRequestError):
        api.generate([], [], None, GenerateConfig(retries=5, retry_interval=0))

    assert attempts == 1


def test_openai_responses_generate_async_does_not_retry_bad_request(monkeypatch) -> None:
    api = _prepare_openai_responses_api()
    attempts = 0

    async def create(**request):
        nonlocal attempts
        attempts += 1
        raise _openai_client_error(openai.BadRequestError, 400)

    async_client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(OpenAIResponsesAPI, 'async_client', property(lambda self: async_client))

    with pytest.raises(openai.BadRequestError):
        asyncio.run(api.generate_async([], [], None, GenerateConfig(retries=5, retry_interval=0)))

    assert attempts == 1


def test_litellm_generate_does_not_retry_bad_request(monkeypatch) -> None:
    litellm = pytest.importorskip('litellm')
    from evalscope.models.litellm_compatible import LiteLLMAPI

    api = object.__new__(LiteLLMAPI)
    api.model_name = 'test-model'
    api.api_key = None
    api.base_url = None
    attempts = 0

    def completion(**request):
        nonlocal attempts
        attempts += 1
        raise litellm.exceptions.BadRequestError('maximum context length is 4096 tokens', 'test-model', 'openai')

    monkeypatch.setattr(litellm, 'completion', completion)

    # LiteLLM's outer handler logs and re-raises; the point is exactly one attempt.
    with pytest.raises(litellm.exceptions.BadRequestError):
        api.generate([], [], None, GenerateConfig(retries=5, retry_interval=0))

    assert attempts == 1


def test_litellm_generate_async_does_not_retry_bad_request(monkeypatch) -> None:
    litellm = pytest.importorskip('litellm')
    from evalscope.models.litellm_compatible import LiteLLMAPI

    api = object.__new__(LiteLLMAPI)
    api.model_name = 'test-model'
    api.api_key = None
    api.base_url = None
    attempts = 0

    async def completion(**request):
        nonlocal attempts
        attempts += 1
        raise litellm.exceptions.BadRequestError('maximum context length is 4096 tokens', 'test-model', 'openai')

    monkeypatch.setattr(litellm, 'acompletion', completion)

    with pytest.raises(litellm.exceptions.BadRequestError):
        asyncio.run(api.generate_async([], [], None, GenerateConfig(retries=5, retry_interval=0)))

    assert attempts == 1
