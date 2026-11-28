"""Trace conversion for externally executed business-agent turns."""

from __future__ import annotations

from typing import Any

from evalscope.api.agent import AgentTrace, EventType
from evalscope.api.model import ModelUsage
from evalscope.benchmarks.project_eval.agent_clients import AgentTurn, as_int
from evalscope.project_eval.schema import EvalTask


def model_usage(usage: dict[str, Any]) -> ModelUsage | None:
    """Convert agent usage counters into a ModelUsage, ignoring malformed values."""
    if not usage:
        return None
    return ModelUsage(
        input_tokens=as_int(usage.get('input_tokens')),
        output_tokens=as_int(usage.get('output_tokens')),
        total_tokens=as_int(usage.get('total_tokens')),
    )


def build_agent_trace(turn: AgentTurn, task: EvalTask) -> AgentTrace:
    """Convert one business-agent turn into EvalScope's structured trace."""
    trace = AgentTrace(
        framework='business_agent',
        trial_id=turn.trace_id,
        total_usage=model_usage(turn.usage),
        events=[],
    )

    step = 0
    trace.add_event(
        step=step,
        type=EventType.RUN_START,
        payload={
            'protocol': task.exec.protocol,
            'trace_available': turn.trace_available,
        },
    )
    for item in turn.steps:
        step += 1
        trace.add_event(
            step=step,
            type=EventType.TOOL_CALL,
            latency_ms=item.elapsed_ms,
            payload={
                'tool': item.name,
                'arguments': item.arguments,
                'confirmed': item.raw.get('confirmed'),
                'phase': item.raw.get('phase'),
                'preview': item.raw.get('preview'),
                'outcome': item.raw.get('outcome'),
            },
        )
        trace.add_event(
            step=step,
            type=EventType.TOOL_RESULT,
            latency_ms=item.elapsed_ms,
            payload={
                'tool': item.name,
                'result': item.result,
                'confirmed': item.raw.get('confirmed'),
                'phase': item.raw.get('phase'),
                'outcome': item.raw.get('outcome'),
            },
        )
    step += 1
    if turn.conversation:
        conversation = list(turn.conversation)
    else:
        conversation = []
        if task.messages:
            conversation.extend({'role': message.role, 'content': message.content} for message in task.messages)
        else:
            if task.system:
                conversation.append({'role': 'system', 'content': task.system})
            conversation.append({'role': 'user', 'content': task.question})
        conversation.append({'role': 'assistant', 'content': turn.reply})
    trace.add_event(
        step=step,
        type=EventType.SUBMIT,
        payload={'reply': turn.reply, 'conversation': conversation},
    )
    step += 1
    trace.add_event(step=step, type=EventType.RUN_END, payload={'trace_available': turn.trace_available})
    return trace


__all__ = ['build_agent_trace', 'model_usage']
