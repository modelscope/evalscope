"""Unit tests for :class:`BridgeTraceRecorder`.

Verify the recorder emits the same :class:`AgentTrace` / :class:`AgentTraceEvent`
shapes the native AgentLoop does, so downstream consumers see one schema
regardless of where the trajectory originated.
"""

import pytest

from evalscope.agent.external.bridge.trace_recorder import BridgeTraceRecorder
from evalscope.api.agent import AgentTrace, EventType
from evalscope.api.messages import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
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
