"""An SSE error event in a 200 stream must surface as a failure, not be silently dropped.

vLLM and SGLang report a failure after the HTTP 200 headers as an SSE event. The
OpenAI SDK's stream iterator raises on the ``{"error": ...}`` form, but the
``{"object": "error", ...}`` form (no ``error`` key) is deserialized (via
``construct``, bypassing validation) into a ``ChatCompletionChunk`` with
``choices=None`` and ``object="error"``. ``collect_stream_response`` used to skip
such chunks, so a partial reply was counted as success. See
modelscope/evalscope#1906.
"""

import asyncio
from typing import Optional

import pytest
from openai.types.chat import ChatCompletionChunk

from evalscope.models.utils.openai import (
    EmptyCompletionError,
    async_collect_stream_response,
    collect_stream_response,
)


def _chunk(content: str, *, finish_reason: Optional[str] = None) -> ChatCompletionChunk:
    """A normal content chunk, validated as the SDK delivers real chunks."""
    return ChatCompletionChunk.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion.chunk',
        'choices': [{'index': 0, 'finish_reason': finish_reason, 'delta': {'content': content}}],
    })


def _usage_chunk() -> ChatCompletionChunk:
    """A usage-only chunk: choices is an empty list (present), not None."""
    return ChatCompletionChunk.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion.chunk',
        'choices': [],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    })


def _error_chunk(message: str = 'internal error') -> ChatCompletionChunk:
    """An SSE ``{"object": "error", ...}`` event as the SDK constructs it.

    ``construct`` bypasses validation, matching the SDK's non-strict path: the
    required ``choices`` field becomes ``None`` and ``object`` is set to
    ``"error"`` despite the ``Literal["chat.completion.chunk"]`` annotation.
    """
    return ChatCompletionChunk.construct(
        id=None,
        created=None,
        model=None,
        object='error',
        choices=None,
        message=message,
        type='internal_error',
    )


def test_sync_error_event_raises_instead_of_returning_partial() -> None:
    # A partial reply followed by an in-band error: the error must win, not be
    # dropped so the partial is counted as success.
    stream = [_chunk('partial'), _error_chunk()]

    with pytest.raises(EmptyCompletionError, match='error event'):
        collect_stream_response(stream)


def test_async_error_event_raises_instead_of_returning_partial() -> None:
    async def stream() -> None:
        yield _chunk('partial')
        yield _error_chunk()

    with pytest.raises(EmptyCompletionError, match='error event'):
        asyncio.run(async_collect_stream_response(stream()))


def test_error_event_with_no_prior_chunks_raises() -> None:
    # An error-only stream raises EmptyCompletionError, not the generic
    # "No valid chat completion chunks" RuntimeError.
    stream = [_error_chunk()]

    with pytest.raises(EmptyCompletionError, match='error event'):
        collect_stream_response(stream)


def test_usage_only_chunk_not_misdetected_as_error() -> None:
    # A usage-only chunk has choices=[] (falsy) but object="chat.completion.chunk";
    # it must still be skipped normally, not treated as an error event.
    stream = [_usage_chunk(), _chunk('hi', finish_reason='stop')]

    completion, _ttft = collect_stream_response(stream)

    assert completion.choices[0].message.content == 'hi'
