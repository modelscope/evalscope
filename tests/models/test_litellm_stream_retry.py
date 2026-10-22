"""LiteLLM retries include stream consumption, not just response creation."""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Optional

import httpx
import litellm
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import ValidationError

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.models.litellm_compatible import LiteLLMAPI


def _chunk(content: str, *, finish_reason: Optional[str] = None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion.chunk',
        'choices': [{'index': 0, 'finish_reason': finish_reason, 'delta': {'content': content}}],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5} if finish_reason else None,
    })


def _mock_provider(
    monkeypatch: pytest.MonkeyPatch,
    response: Callable[[int], ChatCompletion | Iterator[ChatCompletionChunk]],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def completion(**request: Any) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        calls.append(request)
        return response(len(calls))

    async def acompletion(**request: Any) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        result = completion(**request)
        if isinstance(result, ChatCompletion):
            return result

        async def stream() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in result:
                yield chunk

        return stream()

    monkeypatch.setattr(litellm, 'completion', completion)
    monkeypatch.setattr(litellm, 'acompletion', acompletion)
    return calls


def _generate(async_mode: bool, config: GenerateConfig) -> ModelOutput:
    api = LiteLLMAPI(model_name='openai/test-model')
    messages = [ChatMessageUser(content='question')]
    if async_mode:
        return asyncio.run(api.generate_async(messages, [], 'none', config))
    return api.generate(messages, [], 'none', config)


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('error_type', [ConnectionError, httpx.ReadTimeout])
def test_stream_retries_discard_partial_output(
    monkeypatch: pytest.MonkeyPatch, async_mode: bool, error_type: type[Exception]
) -> None:
    failure = error_type('stream interrupted')

    def response(attempt: int) -> Iterator[ChatCompletionChunk]:
        if attempt == 1:
            yield _chunk('discarded partial response')
            raise failure
        yield _chunk('complete ')
        yield _chunk('response', finish_reason='stop')

    calls = _mock_provider(monkeypatch, response)

    output = _generate(async_mode, GenerateConfig(retries=2, retry_interval=0, stream=True))

    assert len(calls) == 2
    assert isinstance(output, ModelOutput)
    assert output.completion == 'complete response'
    assert output.stop_reason == 'stop'
    assert output.usage is not None and output.usage.total_tokens == 5
    assert output.perf_metrics is not None
    assert output.perf_metrics.ttft is not None
    assert output.time is not None
    assert output.perf_metrics.ttft <= output.time


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
def test_stream_retry_exhaustion_raises_the_final_error(
    monkeypatch: pytest.MonkeyPatch, async_mode: bool
) -> None:
    failures = [ConnectionError(f'interrupted attempt {attempt}') for attempt in range(3)]

    def response(attempt: int) -> Iterator[ChatCompletionChunk]:
        yield _chunk('incomplete response')
        raise failures[attempt - 1]

    calls = _mock_provider(monkeypatch, response)

    with pytest.raises(ConnectionError) as caught:
        _generate(async_mode, GenerateConfig(retries=3, retry_interval=0, stream=True))

    assert len(calls) == 3
    assert caught.value is failures[-1]


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
def test_stream_client_error_is_not_retried(monkeypatch: pytest.MonkeyPatch, async_mode: bool) -> None:
    failure = litellm.exceptions.BadRequestError('invalid request', 'test-model', 'openai')

    def response(attempt: int) -> Iterator[ChatCompletionChunk]:
        yield _chunk('incomplete response')
        raise failure

    calls = _mock_provider(monkeypatch, response)

    with pytest.raises(type(failure)) as caught:
        _generate(async_mode, GenerateConfig(retries=3, retry_interval=0, stream=True))

    assert len(calls) == 1
    assert caught.value is failure


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('stream', [False, True])
def test_complete_response_is_still_accepted(
    monkeypatch: pytest.MonkeyPatch, async_mode: bool, stream: bool
) -> None:
    completion = ChatCompletion.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion',
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'answer'}}],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
    })
    calls = _mock_provider(monkeypatch, lambda attempt: completion)

    output = _generate(async_mode, GenerateConfig(retries=3, retry_interval=0, stream=stream))

    assert len(calls) == 1
    assert output.completion == 'answer'
    assert output.usage is not None and output.usage.total_tokens == 5
    assert output.perf_metrics is not None and output.perf_metrics.ttft is None


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('stream', [False, True])
def test_invalid_complete_response_is_validated_without_retry(
    monkeypatch: pytest.MonkeyPatch, async_mode: bool, stream: bool
) -> None:
    completion = ChatCompletion.model_construct(
        id='completion-id', created=1, model='test-model', object='chat.completion', choices=None,
    )
    calls = _mock_provider(monkeypatch, lambda attempt: completion)

    with pytest.raises(ValidationError, match='choices'):
        _generate(async_mode, GenerateConfig(retries=3, retry_interval=0, stream=stream))

    assert len(calls) == 1


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
def test_stream_retry_preserves_provider_finish_reason(
    monkeypatch: pytest.MonkeyPatch, async_mode: bool
) -> None:
    def response(attempt: int) -> Iterator[ChatCompletionChunk]:
        if attempt == 1:
            yield _chunk('discarded partial response')
            raise ConnectionError('connection lost')
        chunk = _chunk('answer')
        chunk.choices[0] = chunk.choices[0].model_copy(update={'finish_reason': 'provider_specific'})
        yield chunk

    calls = _mock_provider(monkeypatch, response)

    output = _generate(async_mode, GenerateConfig(retries=2, retry_interval=0, stream=True))

    assert len(calls) == 2
    assert output.completion == 'answer'
    assert output.stop_reason == 'unknown'
    assert output.metadata == {'finish_reasons': {'0': 'provider_specific'}}


def test_async_stream_cancellation_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def completion(**request: Any) -> AsyncIterator[ChatCompletionChunk]:
            nonlocal calls
            calls += 1

            async def stream() -> AsyncIterator[ChatCompletionChunk]:
                yield _chunk('incomplete response')
                started.set()
                await release.wait()

            return stream()

        monkeypatch.setattr(litellm, 'acompletion', completion)
        api = LiteLLMAPI(model_name='openai/test-model')
        pending = asyncio.create_task(api.generate_async(
            [ChatMessageUser(content='question')], [], 'none', GenerateConfig(retries=3, retry_interval=0, stream=True)
        ))
        await asyncio.wait_for(started.wait(), timeout=2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(run())

    assert calls == 1
