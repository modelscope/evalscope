"""OpenAI Chat Completions SSE synthesizer.

The upstream :meth:`Model.generate_async` call is still non-streaming; once
the :class:`ModelOutput` resolves we slice it into the same
``chat.completion.chunk`` frames the real OpenAI API would emit. This lets
CLI agents that hard-code ``stream=true`` (most OpenAI-protocol clients
do) work against the bridge.

Frame sequence:
    chunk(role='assistant')             # first frame establishes the role
    chunk(delta.reasoning_content=...)  # only when the model returned reasoning
    chunk(delta.content=...) * N        # text body
    chunk(delta.tool_calls=[{index, id, function:{name, arguments=''}}])
    chunk(delta.tool_calls=[{index, function:{arguments='...'}}]) * N
    chunk(finish_reason='stop' | 'tool_calls' | ...)   # last frame; usage iff
                                                       # stream_options.include_usage
    data: [DONE]\\n\\n                                  # OpenAI sentinel
"""

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolCall
from ._sse_common import PING_INTERVAL_S, TEXT_CHUNK, TOOL_INPUT_CHUNK, iter_chunks
from .translate_openai import _split_text_and_reasoning, map_stop_reason_to_openai, unpack_openai_tool_call


def _sse(data: Dict[str, Any]) -> bytes:
    """Encode one OpenAI SSE chunk (data-only, no ``event:`` line)."""
    return f'data: {json.dumps(data, ensure_ascii=False)}\n\n'.encode('utf-8')


def _base_chunk(chunk_id: str, model: str) -> Dict[str, Any]:
    return {
        'id': chunk_id,
        'object': 'chat.completion.chunk',
        'created': int(time.time()),
        'model': model,
        'choices': [{
            'index': 0,
            'delta': {},
            'finish_reason': None,
        }],
    }


async def stream_openai_response(
    generate_task: 'asyncio.Future[ModelOutput]',
    *,
    request_model: Optional[str] = None,
    include_usage: bool = False,
) -> AsyncIterator[bytes]:
    """Yield SSE bytes for the OpenAI chat-completion streaming format."""
    chunk_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'
    model_name = request_model or ''

    # 1. role frame — emitted before awaiting so the SDK initializes its parser
    # and any intermediate proxy starts flushing immediately.
    first = _base_chunk(chunk_id, model_name)
    first['choices'][0]['delta'] = {'role': 'assistant', 'content': ''}
    yield _sse(first)

    # 2. Keep-alive: OpenAI's wire format doesn't have a ping frame, so we send
    # an empty-delta chunk while the upstream model is generating. Most SDKs
    # tolerate it; without it we risk the client timing out before the first
    # real delta arrives.
    output: ModelOutput
    while True:
        try:
            output = await asyncio.wait_for(asyncio.shield(generate_task), timeout=PING_INTERVAL_S)
            break
        except asyncio.TimeoutError:
            ka = _base_chunk(chunk_id, model_name)
            yield _sse(ka)

    if output.error:
        err = _base_chunk(chunk_id, model_name)
        err['choices'][0]['finish_reason'] = 'stop'
        err['error'] = {'type': 'api_error', 'message': output.error}
        yield _sse(err)
        yield b'data: [DONE]\n\n'
        return

    message = output.message if output.choices else None
    if message is not None:
        text, reasoning = _split_text_and_reasoning(message)
        # 3a. reasoning_content (DashScope vendor extension; most clients
        # ignore unknown delta keys, but qwen-thinking-aware clients pick it up).
        if reasoning:
            for chunk in iter_chunks(reasoning, TEXT_CHUNK):
                frame = _base_chunk(chunk_id, model_name)
                frame['choices'][0]['delta'] = {'reasoning_content': chunk}
                yield _sse(frame)
                await asyncio.sleep(0)
        # 3b. main text body
        if text:
            for chunk in iter_chunks(text, TEXT_CHUNK):
                frame = _base_chunk(chunk_id, model_name)
                frame['choices'][0]['delta'] = {'content': chunk}
                yield _sse(frame)
                await asyncio.sleep(0)
        # 3c. tool_calls — emit one announce frame per call (id + name +
        # empty arguments) then stream the JSON arguments in slices. Every
        # frame's tool_calls entry MUST carry its ``index`` so the client
        # can correctly fan out parallel calls.
        for idx, tc in enumerate(message.tool_calls or []):
            async for frame in _emit_tool_call(chunk_id, model_name, tc, idx):
                yield frame

    # 4. final frame — finish_reason set, optional usage block.
    final = _base_chunk(chunk_id, model_name)
    final['choices'][0]['finish_reason'] = map_stop_reason_to_openai(
        output.choices[0].stop_reason if output.choices else 'stop'
    )
    if include_usage:
        usage = output.usage
        final['usage'] = {
            'prompt_tokens': usage.input_tokens if usage else 0,
            'completion_tokens': usage.output_tokens if usage else 0,
            'total_tokens': usage.total_tokens if usage else 0,
        }
    yield _sse(final)

    # 5. OpenAI sentinel.
    yield b'data: [DONE]\n\n'


async def _emit_tool_call(
    chunk_id: str,
    model_name: str,
    tool_call: ToolCall,
    index: int,
) -> AsyncIterator[bytes]:
    name, args = unpack_openai_tool_call(tool_call)
    encoded = json.dumps(args, ensure_ascii=False)
    # Announce frame: function name + id + empty arguments. Subsequent
    # arguments delta frames omit the ``id`` / ``function.name`` fields.
    announce = _base_chunk(chunk_id, model_name)
    announce['choices'][0]['delta'] = {
        'tool_calls': [{
            'index': index,
            'id': tool_call.id,
            'type': 'function',
            'function': {
                'name': name,
                'arguments': '',
            },
        }],
    }
    yield _sse(announce)
    for chunk in iter_chunks(encoded, TOOL_INPUT_CHUNK):
        frame = _base_chunk(chunk_id, model_name)
        frame['choices'][0]['delta'] = {
            'tool_calls': [{
                'index': index,
                'function': {
                    'arguments': chunk,
                },
            }],
        }
        yield _sse(frame)
        await asyncio.sleep(0)
