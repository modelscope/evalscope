"""Unit tests for :class:`BridgeTraceRecorder`.

Verify the recorder emits the same :class:`AgentTrace` / :class:`AgentTraceEvent`
shapes the native AgentLoop does, so downstream consumers see one schema
regardless of where the trajectory originated.
"""

import pytest

from evalscope.agent.external.bridge.trace_recorder import BridgeTraceRecorder
from evalscope.api.agent import AgentTrace, EventType
from evalscope.api.messages import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from evalscope.api.model import ModelOutput, ModelUsage
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.tool import ToolCall, ToolFunction


def _output(text: str, *, tool_calls=None, usage=None) -> ModelOutput:
    msg = ChatMessageAssistant(content=text, tool_calls=tool_calls or None)
    return ModelOutput(
        model='mock',
        choices=[ChatCompletionChoice(message=msg, stop_reason='stop')],
        usage=usage,
    )


def test_first_turn_emits_user_prompt_then_model_generate():
    rec = BridgeTraceRecorder(trial_id='t1', framework='mock', model_name='mock')
    rec.record_run_start(framework='mock', cmd_summary='Mock')
    rec.record_anthropic_turn(
        request_body={'messages': [{'role': 'user', 'content': 'hello'}]},
        output=_output('world', usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5)),
        latency_ms=42.0,
    )
    rec.record_run_end(returncode=0, timed_out=False, wall_time=0.1)
    trace = rec.snapshot()

    assert isinstance(trace, AgentTrace)
    assert trace.framework == 'mock'
    assert trace.trial_id == 't1'
    assert trace.total_usage == ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5)

    types = [ev.type for ev in trace.events]
    assert types == [EventType.RUN_START, EventType.MODEL_GENERATE, EventType.RUN_END]
    gen = trace.events[1]
    assert gen.step == 0
    assert gen.latency_ms == 42.0

    msgs = rec.messages()
    assert [m.role for m in msgs] == ['user', 'assistant']
    assert isinstance(msgs[0], ChatMessageUser)
    assert msgs[0].text == 'hello'
    assert msgs[1].text == 'world'


def test_tool_use_then_tool_result_lands_on_next_step():
    rec = BridgeTraceRecorder(trial_id='t2', framework='mock', model_name='mock')
    # Turn 0: assistant asks for a tool call.
    call = ToolCall(id='call-1', function=ToolFunction(name='lookup', arguments={'q': 'x'}), type='function')
    rec.record_anthropic_turn(
        request_body={'messages': [{'role': 'user', 'content': 'use a tool'}]},
        output=_output('thinking', tool_calls=[call]),
    )
    # Turn 1: bridge sees the tool_result, then asks the model again.
    rec.record_anthropic_turn(
        request_body={
            'messages': [
                {'role': 'user', 'content': 'use a tool'},
                {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'call-1', 'name': 'lookup', 'input': {}}]},
                {
                    'role': 'user',
                    'content': [{'type': 'tool_result', 'tool_use_id': 'call-1', 'content': 'tool-output'}],
                },
            ],
        },
        output=_output('done'),
    )
    trace = rec.snapshot()
    steps_by_type = {(ev.type, ev.step) for ev in trace.events}
    # Turn 0 (step 0): MODEL_GENERATE + TOOL_CALL (no TOOL_RESULT yet)
    assert (EventType.MODEL_GENERATE, 0) in steps_by_type
    assert (EventType.TOOL_CALL, 0) in steps_by_type
    # Turn 1 (step 1): TOOL_RESULT precedes MODEL_GENERATE — TOOL_RESULT
    # is recorded on the new step (matches native AgentLoop ordering).
    assert (EventType.TOOL_RESULT, 1) in steps_by_type
    assert (EventType.MODEL_GENERATE, 1) in steps_by_type

    # Transcript: user + assistant(turn0) + tool + assistant(turn1)
    roles = [m.role for m in rec.messages()]
    assert roles == ['user', 'assistant', 'tool', 'assistant']
    tool_msg = rec.messages()[2]
    assert isinstance(tool_msg, ChatMessageTool)
    assert tool_msg.tool_call_id == 'call-1'
    assert tool_msg.content == 'tool-output'


def test_total_usage_accumulates_across_turns():
    rec = BridgeTraceRecorder(trial_id='t3', framework='mock')
    rec.record_anthropic_turn(
        request_body={'messages': [{'role': 'user', 'content': 'a'}]},
        output=_output('1', usage=ModelUsage(input_tokens=1, output_tokens=2, total_tokens=3)),
    )
    rec.record_anthropic_turn(
        request_body={'messages': [{'role': 'user', 'content': 'a'}]},
        output=_output('2', usage=ModelUsage(input_tokens=4, output_tokens=8, total_tokens=12)),
    )
    assert rec.snapshot().total_usage == ModelUsage(input_tokens=5, output_tokens=10, total_tokens=15)


def test_run_end_records_error_and_returncode():
    rec = BridgeTraceRecorder(trial_id='t4', framework='mock')
    rec.record_run_start(framework='mock', cmd_summary='Mock')
    rec.record_run_end(returncode=137, timed_out=True, wall_time=12.5, error='killed by signal')
    end = [ev for ev in rec.snapshot().events if ev.type == EventType.RUN_END][0]
    assert end.payload['returncode'] == 137
    assert end.payload['timed_out'] is True
    assert end.payload['wall_time'] == pytest.approx(12.5)
    assert end.payload['error'] == 'killed by signal'


def test_responses_first_turn_ingests_instructions_and_system_before_user():
    """Responses path: top-level ``instructions`` + embedded ``role:'system'``
    items must land in the transcript as :class:`ChatMessageSystem` BEFORE the
    first user message, in document order. Captures the bug where codex was
    putting the SWE-bench Pro task in ``instructions`` and the recorder
    silently dropped it (review transcript missing the task description)."""
    rec = BridgeTraceRecorder(trial_id='t5', framework='codex', model_name='qwen3-max')
    rec.record_run_start(framework='codex', cmd_summary='CodexRunner')
    rec.record_responses_turn(
        request_body={
            'instructions': '<SWE-bench Pro task description>',
            'input': [
                {
                    'type': 'message',
                    'role': 'system',
                    'content': [{'type': 'input_text', 'text': 'extra codex system note'}],
                },
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': '<environment_context>'}],
                },
            ],
        },
        output=_output('thinking...', usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12)),
        latency_ms=1.0,
    )
    rec.record_run_end(returncode=0, timed_out=False, wall_time=0.1)

    msgs = rec.messages()
    roles = [m.role for m in msgs]
    assert roles == ['system', 'system', 'user', 'assistant']
    assert isinstance(msgs[0], ChatMessageSystem)
    assert msgs[0].text == '<SWE-bench Pro task description>'
    assert isinstance(msgs[1], ChatMessageSystem)
    assert msgs[1].text == 'extra codex system note'
    assert msgs[2].text == '<environment_context>'


def test_responses_first_turn_ingests_all_user_messages_not_just_first():
    """codex splits the initial prompt across multiple ``role:'user'``
    items (AGENTS.md auto-discovery in one, the actual positional-argv
    task description in another). The transcript must capture **both** —
    stopping at the first user item drops the task description, which
    was the SWE-bench Pro symptom that prompted the rewrite."""
    rec = BridgeTraceRecorder(trial_id='t7', framework='codex')
    rec.record_run_start(framework='codex', cmd_summary='CodexRunner')
    rec.record_responses_turn(
        request_body={
            'instructions': '<codex system prompt>',
            'input': [
                {
                    'type': 'message',
                    'role': 'developer',
                    'content': [{'type': 'input_text', 'text': '<permissions>'}],
                },
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': 'AGENTS.md + env context'}],
                },
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': 'What is 6 * 7?'}],
                },
            ],
        },
        output=_output('42'),
        latency_ms=1.0,
    )
    msgs = rec.messages()
    assert [m.role for m in msgs] == ['system', 'system', 'user', 'user', 'assistant']
    assert msgs[0].text == '<codex system prompt>'
    assert msgs[1].text == '<permissions>'
    assert msgs[2].text == 'AGENTS.md + env context'
    assert msgs[3].text == 'What is 6 * 7?'  # the actual task — must survive


def test_responses_subsequent_turns_do_not_re_ingest_initial_messages():
    """codex re-sends the entire initial setup on every turn. The recorder
    must only ingest it once (on the first turn, gated by ``self._step``)."""
    rec = BridgeTraceRecorder(trial_id='t8', framework='codex')
    initial = {
        'instructions': '<task>',
        'input': [
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'hello'}]},
        ],
    }
    rec.record_responses_turn(request_body=initial, output=_output('ok'), latency_ms=1.0)
    rec.record_responses_turn(
        # Turn 2: full initial setup re-sent + assistant's prior reply + a new tool result.
        request_body={
            'instructions': '<task>',
            'input': [
                {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'hello'}]},
                {'type': 'function_call_output', 'call_id': 'call-1', 'output': 'tool result'},
            ],
        },
        output=_output('done'),
        latency_ms=1.0,
    )
    msgs = rec.messages()
    # Initial setup ingested ONCE: system + user. Then assistant, then
    # tool result, then second assistant.
    roles = [m.role for m in msgs]
    assert roles == ['system', 'user', 'assistant', 'tool', 'assistant']


def test_responses_first_turn_ingests_instructions_when_user_message_empty():
    """When codex sends a non-empty ``instructions`` but the first user
    message is empty (or codex put everything into instructions and only
    sent ``<environment_context>`` as user), the system message still
    surfaces in the transcript so downstream consumers see the task."""
    rec = BridgeTraceRecorder(trial_id='t6', framework='codex')
    rec.record_run_start(framework='codex', cmd_summary='CodexRunner')
    rec.record_responses_turn(
        request_body={
            'instructions': '<full task description>',
            'input': [
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': '<env>'}],
                },
            ],
        },
        output=_output('ok'),
        latency_ms=1.0,
    )
    msgs = rec.messages()
    assert [m.role for m in msgs] == ['system', 'user', 'assistant']
    assert msgs[0].text == '<full task description>'
