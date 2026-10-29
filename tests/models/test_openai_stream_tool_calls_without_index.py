import asyncio
from typing import Any, AsyncIterator, List, Optional

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from evalscope.models.utils.openai import async_collect_stream_response, collect_stream_response


def _call(
    index: Optional[int],
    call_id: Optional[str] = None,
    name: Optional[str] = None,
    arguments: Optional[str] = None,
) -> ChoiceDeltaToolCall:
    # ``model_construct`` keeps ``index`` unset, the way Gemini's OpenAI endpoint omits it.
    function = ChoiceDeltaToolCallFunction(name=name, arguments=arguments)
    if index is None:
        return ChoiceDeltaToolCall.model_construct(id=call_id, type='function', function=function)
    return ChoiceDeltaToolCall(index=index, id=call_id, type='function', function=function)


def _chunk(calls: List[ChoiceDeltaToolCall], finish_reason: Optional[str] = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id='chatcmpl-1',
        created=1,
        model='test-model',
        object='chat.completion.chunk',
        choices=[Choice(index=0, delta=ChoiceDelta(tool_calls=calls or None), finish_reason=finish_reason)],
    )


def _collect(chunks: List[ChatCompletionChunk], asynchronous: bool) -> Any:
    if not asynchronous:
        completion, _ = collect_stream_response(iter(chunks), request_start=0.0)
        return completion

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in chunks:
            yield chunk

    completion, _ = asyncio.run(async_collect_stream_response(stream(), request_start=0.0))
    return completion


def _calls(completion: Any) -> List[tuple]:
    message = completion.choices[0].message
    return [(tc.id, tc.function.name, tc.function.arguments) for tc in message.tool_calls]


@pytest.mark.parametrize('asynchronous', [False, True])
def test_parallel_calls_without_index_stay_separate(asynchronous: bool) -> None:
    chunks = [
        _chunk([_call(None, 'call_a', 'get_weather', '{"city": "Paris"}')]),
        _chunk([_call(None, 'call_b', 'get_time', '{"tz": "UTC"}')]),
        _chunk([], finish_reason='tool_calls'),
    ]

    completion = _collect(chunks, asynchronous)

    assert _calls(completion) == [
        ('call_a', 'get_weather', '{"city": "Paris"}'),
        ('call_b', 'get_time', '{"tz": "UTC"}'),
    ]


@pytest.mark.parametrize('asynchronous', [False, True])
def test_call_without_index_continues_across_argument_chunks(asynchronous: bool) -> None:
    chunks = [
        _chunk([_call(None, 'call_a', 'get_weather', '{"city": ')]),
        _chunk([_call(None, None, None, '"Paris"}')]),
        _chunk([_call(None, 'call_b', 'get_time', '{}')]),
        _chunk([], finish_reason='tool_calls'),
    ]

    completion = _collect(chunks, asynchronous)

    assert _calls(completion) == [
        ('call_a', 'get_weather', '{"city": "Paris"}'),
        ('call_b', 'get_time', '{}'),
    ]


@pytest.mark.parametrize('asynchronous', [False, True])
def test_calls_with_index_are_merged_by_index(asynchronous: bool) -> None:
    chunks = [
        _chunk([_call(0, 'call_a', 'get_weather', '{"city": ')]),
        _chunk([_call(1, 'call_b', 'get_time', '{}')]),
        _chunk([_call(0, None, None, '"Paris"}')]),
        _chunk([], finish_reason='tool_calls'),
    ]

    completion = _collect(chunks, asynchronous)

    assert _calls(completion) == [
        ('call_a', 'get_weather', '{"city": "Paris"}'),
        ('call_b', 'get_time', '{}'),
    ]
