"""Strict Responses API SSE event-sequence oracle.

Codex CLI deduplicates events by ``sequence_number`` and aborts on
out-of-order or missing types. This test exercises
:func:`evalscope.agent.external.bridge.sse_responses.stream_responses_payload`
directly (no HTTP round-trip) so it runs fast and pinpoints any event
sequence regression independent of the server / model layers.

Reference: inspect_ai's agent bridge proxy.py implements the same
sequence; PR2 mirrors it field-for-field.
"""

import json
import pytest
from typing import Any, Dict, List, Tuple

from evalscope.agent.external.bridge.sse_responses import stream_responses_payload
from evalscope.utils.function_utils import AsyncioLoopRunner


@pytest.fixture(autouse=True)
def _release_loop():
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def _drain(payload: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Run the async iterator to completion and parse each SSE frame back
    into ``(event_type, data_dict)`` pairs."""

    async def _go() -> List[Tuple[str, Dict[str, Any]]]:
        out: List[Tuple[str, Dict[str, Any]]] = []
        async for chunk in stream_responses_payload(payload):
            text = chunk.decode('utf-8')
            # SSE frame: "event: <type>\ndata: <json>\n\n"
            lines = text.rstrip('\n').split('\n')
            assert lines[0].startswith('event: '), f'malformed frame: {text!r}'
            assert lines[1].startswith('data: '), f'malformed frame: {text!r}'
            event = lines[0][len('event: '):]
            data = json.loads(lines[1][len('data: '):])
            out.append((event, data))
        return out

    return AsyncioLoopRunner.run(_go())


def _base_payload(output_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        'id': 'resp_test_abcdef',
        'object': 'response',
        'created_at': 1234567890,
        'status': 'completed',
        'model': 'mock-responses',
        'output': output_items,
        'usage': {
            'input_tokens': 7,
            'output_tokens': 11,
            'total_tokens': 18
        },
    }


def _assert_sequence_numbers_strictly_monotonic(frames):
    seqs = [data['sequence_number'] for _, data in frames]
    assert seqs == list(range(1, len(seqs) + 1)), (
        f'sequence_numbers not strictly monotonic from 1: {seqs}'
    )


def test_message_only_payload_emits_expected_event_sequence():
    """Single assistant text message — content_part bracketing + text deltas + done."""
    payload = _base_payload([{
        'type': 'message',
        'id': 'msg_001',
        'role': 'assistant',
        'content': [{
            'type': 'output_text',
            'text': 'hello world!'
        }],
        'status': 'completed',
    }])

    frames = _drain(payload)
    events = [ev for ev, _ in frames]

    assert events == [
        'response.created',
        'response.in_progress',
        'response.output_item.added',
        'response.content_part.added',
        'response.output_text.delta',  # 'hello world!' < 48 chars → single delta
        'response.output_text.done',
        'response.content_part.done',
        'response.output_item.done',
        'response.completed',
    ]
    _assert_sequence_numbers_strictly_monotonic(frames)

    # created / in_progress share the same response shell, status='in_progress'.
    assert frames[0][1]['response']['status'] == 'in_progress'
    assert frames[1][1]['response']['status'] == 'in_progress'
    # completed flips to 'completed'.
    assert frames[-1][1]['response']['status'] == 'completed'

    # The 'added' content_part has text cleared; the 'done' part has it back.
    added_part = frames[3][1]['part']
    done_part = frames[6][1]['part']
    assert added_part['text'] == ''
    assert done_part['text'] == 'hello world!'

    # The 'output_text.done' carries the full text + empty logprobs for shape parity.
    done_text = frames[5][1]
    assert done_text['text'] == 'hello world!'
    assert done_text['logprobs'] == []

    # Every body frame references the item_id so codex can correlate.
    for ev, data in frames[2:-1]:
        if ev != 'response.output_item.added' and ev != 'response.output_item.done':
            assert data['item_id'] == 'msg_001', f'{ev} missing item_id'


def test_function_call_payload_emits_arguments_delta_then_done():
    """Function call → arguments chunked 32-char at a time, then done."""
    long_args = json.dumps({'query': 'a' * 80, 'limit': 5}, ensure_ascii=False)
    payload = _base_payload([{
        'type': 'function_call',
        'id': 'fc_001',
        'call_id': 'call_xyz',
        'name': 'lookup',
        'arguments': long_args,
        'status': 'completed',
    }])

    frames = _drain(payload)
    events = [ev for ev, _ in frames]

    # Compute expected number of delta frames (32-char chunks)
    n_delta = (len(long_args) + 31) // 32

    assert events[:3] == [
        'response.created',
        'response.in_progress',
        'response.output_item.added',
    ]
    assert events[3:3 + n_delta] == ['response.function_call_arguments.delta'] * n_delta
    assert events[3 + n_delta:] == [
        'response.function_call_arguments.done',
        'response.output_item.done',
        'response.completed',
    ]
    _assert_sequence_numbers_strictly_monotonic(frames)

    # Reassembling delta chunks must reproduce the full arguments JSON.
    reassembled = ''.join(
        data['delta'] for ev, data in frames if ev == 'response.function_call_arguments.delta'
    )
    assert reassembled == long_args

    # The 'done' frame echoes the full arguments string.
    done_frame = next(data for ev, data in frames if ev == 'response.function_call_arguments.done')
    assert done_frame['arguments'] == long_args


def test_mixed_payload_reasoning_message_function_call_in_order():
    """Reasoning + message + function_call interleaved → full event matrix."""
    payload = _base_payload([
        {
            'type': 'reasoning',
            'id': 'rs_001',
            'summary': [{
                'type': 'summary_text',
                'text': 'I should call the tool then summarise.'
            }],
        },
        {
            'type': 'message',
            'id': 'msg_001',
            'role': 'assistant',
            'content': [{
                'type': 'output_text',
                'text': 'Calling tool now.'
            }],
            'status': 'completed',
        },
        {
            'type': 'function_call',
            'id': 'fc_001',
            'call_id': 'call_a',
            'name': 'lookup',
            'arguments': '{"q":"x"}',
            'status': 'completed',
        },
    ])

    frames = _drain(payload)
    events = [ev for ev, _ in frames]

    # Top-level frame: created → in_progress.
    assert events[0:2] == ['response.created', 'response.in_progress']

    # Reasoning bracket: output_item.added → summary_part.added → summary_text.delta+ → done → summary_part.done → output_item.done.
    reasoning_slice = events[2:9]
    assert reasoning_slice[0] == 'response.output_item.added'
    assert reasoning_slice[1] == 'response.reasoning_summary_part.added'
    assert reasoning_slice[2] == 'response.reasoning_summary_text.delta'
    assert reasoning_slice[3] == 'response.reasoning_summary_text.done'
    assert reasoning_slice[4] == 'response.reasoning_summary_part.done'
    assert reasoning_slice[5] == 'response.output_item.done'

    # Final frame: completed.
    assert events[-1] == 'response.completed'

    # No 'data: [DONE]' sentinel — Responses uses 'response.completed' as terminator.
    assert all('[DONE]' not in ev for ev in events)

    # Sequence numbers strictly monotonic across the whole stream.
    _assert_sequence_numbers_strictly_monotonic(frames)

    # Output_index advances per top-level item (0,1,2).
    item_indexes = [data['output_index'] for ev, data in frames if ev == 'response.output_item.added']
    assert item_indexes == [0, 1, 2]
