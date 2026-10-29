"""Tool function names are string deltas, just like their JSON arguments."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from openai.types.chat import ChatCompletionChunk

from evalscope.models.utils.openai import (
    async_collect_stream_response,
    chat_choices_from_openai,
    collect_stream_response,
    model_output_from_openai,
)


def _chunk(tool_calls: list[dict[str, Any]], *, finished: bool = False) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion.chunk',
        'choices': [{
            'index': 0,
            'delta': {'tool_calls': tool_calls} if tool_calls else {},
            'finish_reason': 'tool_calls' if finished else None,
        }],
    })


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('fragmented', [False, True], ids=['complete_name', 'fragmented_name'])
@pytest.mark.parametrize('parallel', [False, True], ids=['single_tool', 'parallel_tools'])
def test_streaming_tool_names_are_accumulated(
    asynchronous: bool, fragmented: bool, parallel: bool
) -> None:
    chunks = []
    names = [['get_', 'wea', 'ther']] if fragmented else [['get_weather', None, None]]
    if parallel:
        names.append(['get_', 'ti', 'me'] if fragmented else ['get_time', None, None])

    expected_names = ['get_weather', 'get_time'] if parallel else ['get_weather']
    arguments = ['{"city":', '"Paris"', '}']
    for step in range(3):
        tool_calls = []
        for index, parts in enumerate(names):
            function: dict[str, Any] = {'arguments': arguments[step]}
            if parts[step] is not None:
                function['name'] = parts[step]
            tool_call: dict[str, Any] = {'index': index, 'function': function}
            if step == 0:
                tool_call.update(id=f'call_{index}', type='function')
            tool_calls.append(tool_call)
        # Interleave parallel fragments in reverse order after their first chunk.
        if step:
            tool_calls.reverse()
        chunk = _chunk(tool_calls)
        chunks.append(chunk)
    chunks.append(_chunk([], finished=True))

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in chunks:
            yield chunk

    if asynchronous:
        completion, _ = asyncio.run(async_collect_stream_response(stream()))
    else:
        completion, _ = collect_stream_response(chunks)

    tool_calls = completion.choices[0].message.tool_calls
    assert tool_calls is not None
    assert [call.function.name for call in tool_calls] == expected_names
    assert [call.function.arguments for call in tool_calls] == ['{"city":"Paris"}'] * len(expected_names)
    assert [call.id for call in tool_calls] == [f'call_{index}' for index in range(len(expected_names))]

    output = model_output_from_openai(completion, chat_choices_from_openai(completion, tools=[]))
    assert [call.function.name for call in output.message.tool_calls] == expected_names
    assert [call.function.arguments for call in output.message.tool_calls] == [
        {'city': 'Paris'} for _ in expected_names
    ]
