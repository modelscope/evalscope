"""OpenAI Responses API SSE synthesizer.

Pre-resolve pattern: the upstream :meth:`Model.generate_async` call is
non-streaming; once the full :class:`ModelOutput` resolves, the bridge
renders it into a Responses payload dict (see
:func:`translate_responses.model_output_to_responses_payload`) and this
synthesizer slices that dict into the SSE event sequence codex expects.

Event sequence (mirrors inspect_ai's agent bridge):

.. code-block:: text

    response.created                          (response payload, status='in_progress')
    response.in_progress                      (same shell)
    for each output_item:
        response.output_item.added
        if message:
            for each content_part:
                response.content_part.added   (output_text with text cleared)
                response.output_text.delta * N
                response.output_text.done
                response.content_part.done    (full part)
        elif function_call:
            response.function_call_arguments.delta * N  (32-char chunks)
            response.function_call_arguments.done
        elif reasoning:
            for each summary part:
                response.reasoning_summary_part.added
                response.reasoning_summary_text.delta * N
                response.reasoning_summary_text.done
                response.reasoning_summary_part.done
        response.output_item.done
    response.completed                        (response payload, status='completed')

Notes:

* No ``data: [DONE]`` sentinel — ``response.completed`` is the terminator.
* Every event payload carries a strictly-monotonic ``sequence_number``
  (codex deduplicates by it and aborts on out-of-order).
* ``function_call_arguments`` chunks are 32 chars (inspect's empirical
  value); message text uses the shared :data:`TEXT_CHUNK` (48).
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List

from ._sse_common import TEXT_CHUNK, iter_chunks

#: Per-frame slice width for ``function_call_arguments.delta``. Smaller
#: than ``TEXT_CHUNK`` because codex parses partial JSON on each frame
#: and benefits from finer granularity. Value matches inspect_ai's
#: agent bridge for cross-implementation consistency.
_FUNCTION_ARG_CHUNK = 32


async def stream_responses_payload(payload: Dict[str, Any]) -> AsyncIterator[bytes]:
    """Yield SSE bytes for the Responses API event sequence.

    ``payload`` is the dict from
    :func:`translate_responses.model_output_to_responses_payload`. The
    synthesizer mutates only local copies (``status`` field is overridden
    on the ``in_progress`` / ``completed`` frames); the caller's payload
    is not modified.
    """
    seq = [0]

    def _frame(event: str, data: Dict[str, Any]) -> bytes:
        seq[0] += 1
        body = {**data, 'sequence_number': seq[0], 'type': event}
        return f'event: {event}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n'.encode('utf-8')

    in_progress_resp = {**payload, 'status': 'in_progress'}
    yield _frame('response.created', {'response': in_progress_resp})
    yield _frame('response.in_progress', {'response': in_progress_resp})

    for output_index, item in enumerate(payload.get('output') or []):
        async for frame in _emit_output_item(item, output_index, _frame):
            yield frame

    completed_resp = {**payload, 'status': 'completed'}
    yield _frame('response.completed', {'response': completed_resp})


async def _emit_output_item(
    item: Dict[str, Any],
    output_index: int,
    frame_fn,
) -> AsyncIterator[bytes]:
    item_id = item.get('id') or f'item_{output_index}'
    itype = item.get('type')

    added_item = dict(item)
    if 'status' in added_item:
        added_item['status'] = 'in_progress'
    yield frame_fn('response.output_item.added', {
        'item': added_item,
        'output_index': output_index,
    })

    if itype == 'message':
        async for frame in _emit_message_content(item, item_id, output_index, frame_fn):
            yield frame
    elif itype == 'function_call':
        async for frame in _emit_function_call(item, item_id, output_index, frame_fn):
            yield frame
    elif itype == 'reasoning':
        async for frame in _emit_reasoning(item, item_id, output_index, frame_fn):
            yield frame
    # Unknown item types: just bracket with added/done, no body events.

    yield frame_fn('response.output_item.done', {
        'item': item,
        'output_index': output_index,
    })


async def _emit_message_content(
    item: Dict[str, Any],
    item_id: str,
    output_index: int,
    frame_fn,
) -> AsyncIterator[bytes]:
    parts: List[Dict[str, Any]] = item.get('content') or []
    for content_index, cpart in enumerate(parts):
        ctype = cpart.get('type')
        part_for_added = dict(cpart)
        if ctype == 'output_text':
            part_for_added['text'] = ''
        elif ctype == 'refusal':
            part_for_added['refusal'] = ''
        yield frame_fn(
            'response.content_part.added', {
                'item_id': item_id,
                'output_index': output_index,
                'content_index': content_index,
                'part': part_for_added,
            }
        )

        if ctype == 'output_text':
            text = cpart.get('text', '') or ''
            for chunk in iter_chunks(text, TEXT_CHUNK):
                yield frame_fn(
                    'response.output_text.delta', {
                        'item_id': item_id,
                        'output_index': output_index,
                        'content_index': content_index,
                        'delta': chunk,
                        'logprobs': [],
                    }
                )
                await asyncio.sleep(0)
            yield frame_fn(
                'response.output_text.done', {
                    'item_id': item_id,
                    'output_index': output_index,
                    'content_index': content_index,
                    'text': text,
                    'logprobs': [],
                }
            )
        elif ctype == 'refusal':
            refusal = cpart.get('refusal', '') or ''
            for chunk in iter_chunks(refusal, TEXT_CHUNK):
                yield frame_fn(
                    'response.refusal.delta', {
                        'item_id': item_id,
                        'output_index': output_index,
                        'content_index': content_index,
                        'delta': chunk,
                    }
                )
                await asyncio.sleep(0)
            yield frame_fn(
                'response.refusal.done', {
                    'item_id': item_id,
                    'output_index': output_index,
                    'content_index': content_index,
                    'refusal': refusal,
                }
            )

        yield frame_fn(
            'response.content_part.done', {
                'item_id': item_id,
                'output_index': output_index,
                'content_index': content_index,
                'part': cpart,
            }
        )


async def _emit_function_call(
    item: Dict[str, Any],
    item_id: str,
    output_index: int,
    frame_fn,
) -> AsyncIterator[bytes]:
    args = item.get('arguments', '') or ''
    for chunk in iter_chunks(args, _FUNCTION_ARG_CHUNK):
        yield frame_fn(
            'response.function_call_arguments.delta', {
                'item_id': item_id,
                'output_index': output_index,
                'delta': chunk,
            }
        )
        await asyncio.sleep(0)
    yield frame_fn(
        'response.function_call_arguments.done', {
            'item_id': item_id,
            'output_index': output_index,
            'arguments': args,
        }
    )


async def _emit_reasoning(
    item: Dict[str, Any],
    item_id: str,
    output_index: int,
    frame_fn,
) -> AsyncIterator[bytes]:
    for summary_index, summary in enumerate(item.get('summary') or []):
        yield frame_fn(
            'response.reasoning_summary_part.added', {
                'item_id': item_id,
                'output_index': output_index,
                'summary_index': summary_index,
                'part': summary,
            }
        )
        if summary.get('type') == 'summary_text':
            text = summary.get('text', '') or ''
            for chunk in iter_chunks(text, TEXT_CHUNK):
                yield frame_fn(
                    'response.reasoning_summary_text.delta', {
                        'item_id': item_id,
                        'output_index': output_index,
                        'summary_index': summary_index,
                        'delta': chunk,
                    }
                )
                await asyncio.sleep(0)
            yield frame_fn(
                'response.reasoning_summary_text.done', {
                    'item_id': item_id,
                    'output_index': output_index,
                    'summary_index': summary_index,
                    'text': text,
                }
            )
        yield frame_fn(
            'response.reasoning_summary_part.done', {
                'item_id': item_id,
                'output_index': output_index,
                'summary_index': summary_index,
                'part': summary,
            }
        )


__all__ = ['stream_responses_payload']
