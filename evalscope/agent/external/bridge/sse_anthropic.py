"""Anthropic Messages SSE event synthesizer.

Implements the synthetic-stream pattern from inspect_ai's agent bridge: the
underlying ``Model.generate_async`` call is still non-streaming; once the
full :class:`ModelOutput` is available we slice it into the same SSE event
sequence the real Anthropic API would emit.  This is the minimum required
to make CLI agents (claude-code, codex) that hard-code ``stream=true``
work against the bridge.

Event sequence (per Anthropic SSE spec):
    message_start
    [ping] (keep-alive every 5s while waiting for the model)
    for each content block:
        content_block_start
        content_block_delta * N    (text_delta / input_json_delta / thinking_delta)
        content_block_stop
    message_delta                  (stop_reason + usage)
    message_stop
"""

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from evalscope.api.model import ModelOutput
from ._sse_common import PING_INTERVAL_S, TEXT_CHUNK, TOOL_INPUT_CHUNK, iter_chunks
from .translate_anthropic import map_stop_reason_to_anthropic, unpack_tool_call


def _sse(event_type: str, data: Dict[str, Any]) -> bytes:
    """Encode one SSE event (Anthropic uses ``event:`` + ``data:`` lines)."""
    return f'event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'.encode('utf-8')


async def stream_anthropic_response(
    generate_task: 'asyncio.Future[ModelOutput]',
    *,
    request_model: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """Yield SSE bytes for the Anthropic streaming format.

    ``generate_task`` is awaited once; the result is then re-emitted as a
    synthetic stream.  ``ping`` events are interleaved while waiting so the
    client (and any intermediate proxy) sees bytes immediately.
    """
    message_id = f'msg_{uuid.uuid4().hex[:24]}'
    model_name = request_model or ''

    # 1. message_start — sent BEFORE awaiting completion so the SDK initializes
    # its parser and intermediate proxies don't time out.
    yield _sse(
        'message_start', {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'content': [],
                'model': model_name,
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'output_tokens': 0
                },
            },
        }
    )

    # 2. Keep-alive pings while waiting for the underlying generation.
    # ``shield`` prevents the wait_for timeout from cancelling the real call.
    output: ModelOutput
    while True:
        try:
            output = await asyncio.wait_for(asyncio.shield(generate_task), timeout=PING_INTERVAL_S)
            break
        except asyncio.TimeoutError:
            yield _sse('ping', {'type': 'ping'})

    if output.error:
        yield _sse('error', {
            'type': 'error',
            'error': {
                'type': 'api_error',
                'message': output.error
            },
        })
        return

    # 3. Per content block in the assistant message.
    message = output.message if output.choices else None
    index = 0
    if message is not None:
        # 3a. text content (always emit at least one text block — Anthropic
        # SDK can choke on a message with zero blocks).
        text = message.text or ''
        if text or not (message.tool_calls or []):
            async for chunk in _emit_text_block(text, index):
                yield chunk
            index += 1
        # 3b. tool_use content
        for tc in message.tool_calls or []:
            async for chunk in _emit_tool_use_block(tc, index):
                yield chunk
            index += 1

    # 4. message_delta — stop reason + cumulative usage.
    stop_reason = map_stop_reason_to_anthropic(output.choices[0].stop_reason if output.choices else 'stop')
    usage = output.usage
    delta_payload: Dict[str, Any] = {
        'type': 'message_delta',
        'delta': {
            'stop_reason': stop_reason,
            'stop_sequence': None
        },
        'usage': {
            'input_tokens': usage.input_tokens if usage else 0,
            'output_tokens': usage.output_tokens if usage else 0,
        },
    }
    yield _sse('message_delta', delta_payload)

    # 5. message_stop — terminates the stream (no [DONE] sentinel for Anthropic).
    yield _sse('message_stop', {'type': 'message_stop'})


async def _emit_text_block(text: str, index: int) -> AsyncIterator[bytes]:
    yield _sse(
        'content_block_start', {
            'type': 'content_block_start',
            'index': index,
            'content_block': {
                'type': 'text',
                'text': ''
            },
        }
    )
    for chunk in iter_chunks(text, TEXT_CHUNK):
        yield _sse(
            'content_block_delta', {
                'type': 'content_block_delta',
                'index': index,
                'delta': {
                    'type': 'text_delta',
                    'text': chunk
                },
            }
        )
        await asyncio.sleep(0)
    yield _sse('content_block_stop', {
        'type': 'content_block_stop',
        'index': index,
    })


async def _emit_tool_use_block(tool_call: Any, index: int) -> AsyncIterator[bytes]:
    name, args = unpack_tool_call(tool_call)
    yield _sse(
        'content_block_start', {
            'type': 'content_block_start',
            'index': index,
            'content_block': {
                'type': 'tool_use',
                'id': tool_call.id,
                'name': name,
                'input': {},
            },
        }
    )
    encoded = json.dumps(args, ensure_ascii=False)
    for chunk in iter_chunks(encoded, TOOL_INPUT_CHUNK):
        yield _sse(
            'content_block_delta', {
                'type': 'content_block_delta',
                'index': index,
                'delta': {
                    'type': 'input_json_delta',
                    'partial_json': chunk
                },
            }
        )
        await asyncio.sleep(0)
    yield _sse('content_block_stop', {
        'type': 'content_block_stop',
        'index': index,
    })
