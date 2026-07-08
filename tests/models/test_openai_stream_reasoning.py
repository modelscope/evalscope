import asyncio
import pytest
from openai.types.chat import ChatCompletionChunk

from evalscope.models.utils.openai import async_collect_stream_response, collect_stream_response


def _chunk(*, reasoning_content=None, reasoning=None, content=None, finish_reason=None):
    return ChatCompletionChunk.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion.chunk',
        'choices': [{
            'index': 0,
            'finish_reason': finish_reason,
            'delta': {'content': content, 'reasoning_content': reasoning_content, 'reasoning': reasoning},
        }],
    })


def test_collects_reasoning_and_measures_ttft_from_reasoning(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr('evalscope.models.utils.openai.time.monotonic', lambda: clock[0])

    def stream():
        clock[0] = 10.25
        yield _chunk(reasoning='thinking')
        clock[0] = 11.0
        yield _chunk(content='answer', finish_reason='stop')

    completion, ttft = collect_stream_response(stream(), request_start=10.0)

    assert completion.choices[0].message.content == 'answer'
    assert (getattr(completion.choices[0].message, 'reasoning_content', None), ttft) == (
        'thinking', pytest.approx(0.25)
    )


def test_async_collects_reasoning_and_measures_ttft_from_reasoning(monkeypatch):
    clock = [20.0]
    monkeypatch.setattr('evalscope.models.utils.openai.time.monotonic', lambda: clock[0])

    async def stream():
        clock[0] = 20.25
        yield _chunk(reasoning='thinking')
        clock[0] = 21.0
        yield _chunk(content='answer', finish_reason='stop')

    completion, ttft = asyncio.run(async_collect_stream_response(stream(), request_start=20.0))

    assert completion.choices[0].message.content == 'answer'
    assert (getattr(completion.choices[0].message, 'reasoning_content', None), ttft) == (
        'thinking', pytest.approx(0.25)
    )


def test_falls_back_to_reasoning_when_reasoning_content_is_empty(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr('evalscope.models.utils.openai.time.monotonic', lambda: clock[0])

    def stream():
        clock[0] = 10.25
        yield _chunk(reasoning_content='', reasoning='thinking')
        clock[0] = 11.0
        yield _chunk(content='answer', finish_reason='stop')

    completion, ttft = collect_stream_response(stream(), request_start=10.0)

    assert completion.choices[0].message.content == 'answer'
    assert getattr(completion.choices[0].message, 'reasoning_content', None) == 'thinking'
    assert ttft == pytest.approx(0.25)


def test_async_falls_back_to_reasoning_when_reasoning_content_is_empty(monkeypatch):
    clock = [20.0]
    monkeypatch.setattr('evalscope.models.utils.openai.time.monotonic', lambda: clock[0])

    async def stream():
        clock[0] = 20.25
        yield _chunk(reasoning_content='', reasoning='thinking')
        clock[0] = 21.0
        yield _chunk(content='answer', finish_reason='stop')

    completion, ttft = asyncio.run(async_collect_stream_response(stream(), request_start=20.0))

    assert completion.choices[0].message.content == 'answer'
    assert getattr(completion.choices[0].message, 'reasoning_content', None) == 'thinking'
    assert ttft == pytest.approx(0.25)
