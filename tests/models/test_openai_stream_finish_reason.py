import asyncio
from collections.abc import AsyncIterator
from typing import List, Optional, Tuple

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from evalscope.models.utils.openai import async_collect_stream_response, collect_stream_response


def _packed_chunk(choices: List[Tuple[int, Optional[str], Optional[str]]]) -> ChatCompletionChunk:
    """Build a chunk packing several choices (as servers do for n > 1).

    ``choices`` is a list of ``(index, content, finish_reason)`` tuples.
    """
    return ChatCompletionChunk.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion.chunk',
        'choices': [
            {
                'index': index,
                'finish_reason': finish_reason,
                'delta': {'content': content},
            } for index, content, finish_reason in choices
        ],
    })


def test_sync_finish_reason_restored_for_every_packed_choice() -> None:
    stream = [
        _packed_chunk([(0, 'first', None), (1, 'second', None)]),
        _packed_chunk([(0, None, 'stop'), (1, None, 'length')]),
    ]

    completion, _ttft = collect_stream_response(stream)

    finish_reasons = {choice.index: choice.finish_reason for choice in completion.choices}
    assert finish_reasons == {0: 'stop', 1: 'length'}


def test_async_finish_reason_restored_for_every_packed_choice() -> None:
    async def stream() -> None:
        yield _packed_chunk([(0, 'first', None), (1, 'second', None)])
        yield _packed_chunk([(0, None, 'tool_calls'), (1, None, 'content_filter')])

    completion, _ttft = asyncio.run(async_collect_stream_response(stream()))

    finish_reasons = {choice.index: choice.finish_reason for choice in completion.choices}
    assert finish_reasons == {0: 'tool_calls', 1: 'content_filter'}


def test_sync_finish_reason_matches_choice_beyond_first_position() -> None:
    # the matching choice is not the first element of chunk.choices
    stream = [
        _packed_chunk([(1, 'second', None), (0, 'first', None)]),
        _packed_chunk([(1, None, 'length'), (0, None, 'stop')]),
    ]

    completion, _ttft = collect_stream_response(stream)

    finish_reasons = {choice.index: choice.finish_reason for choice in completion.choices}
    assert finish_reasons == {0: 'stop', 1: 'length'}


def _collect(chunks: List[ChatCompletionChunk], asynchronous: bool) -> ChatCompletion:
    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in chunks:
            yield chunk

    if asynchronous:
        completion, _ = asyncio.run(async_collect_stream_response(stream()))
    else:
        completion, _ = collect_stream_response(chunks)
    return completion


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('content', ['unfinished', None], ids=['content', 'metadata_only'])
def test_every_observed_choice_requires_a_finish_reason(asynchronous: bool, content: Optional[str]) -> None:
    chunks = [_packed_chunk([(0, 'complete', 'stop'), (1, content, None)])]

    with pytest.raises(RuntimeError, match='choices.*1'):
        _collect(chunks, asynchronous)


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
def test_trailing_metadata_does_not_erase_finish_reasons(asynchronous: bool) -> None:
    chunks = [
        _packed_chunk([(0, 'first', 'length'), (1, None, 'content_filter')]),
        _packed_chunk([(0, None, None), (1, None, None)]),
    ]

    completion = _collect(chunks, asynchronous)

    assert {choice.index: choice.finish_reason for choice in completion.choices} == {0: 'length', 1: 'content_filter'}
