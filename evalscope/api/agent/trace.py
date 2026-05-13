"""Agent trace event model.

Replaces the legacy ``TaskState._trajectory`` / ``TrajectoryStep`` mechanism.
The trace is persisted as part of ``ReviewResult.agent_trace`` (no separate
file) and is ``None`` for non-agent benchmarks.
"""

import time
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """Canonical event kinds emitted by the AgentLoop."""

    MODEL_GENERATE = 'model_generate'
    """A round of ``model.generate`` returned."""

    TOOL_CALL = 'tool_call'
    """Strategy parsed one tool call out of the model output."""

    TOOL_RESULT = 'tool_result'
    """ToolExecutor finished executing a tool call."""

    ENV_EXEC = 'env_exec'
    """Environment executed a raw command (bash, etc.)."""

    ERROR = 'error'
    """Parse error / tool error / timeout etc."""

    SUBMIT = 'submit'
    """Strategy declared the task finished (final_answer / submit tool)."""


class AgentTraceEvent(BaseModel):
    """Single structured event in an agent trajectory.

    Kept small & JSON-serializable so it can ride along on every
    ``ReviewResult`` without blowing up cache files.
    """

    step: int
    """0-based loop iteration when this event was emitted."""

    timestamp: float = Field(default_factory=time.time)
    """``time.time()`` when recorded (seconds since epoch)."""

    type: EventType
    """Event kind (see :class:`EventType`)."""

    message_id: Optional[str] = None
    """ID of the ``ChatMessage`` this event was emitted for.

    - ``model_generate`` / ``tool_call`` / parse ``error`` / ``submit`` (final_answer)
      → id of the assistant message just produced.
    - ``tool_result`` → id of the tool/user observation message just appended.
    - ``submit`` (post_execution) → id of the last observation message in this step.
    - Loop-level events with no message (e.g. ``max_steps_exceeded``) leave this ``None``.
    """

    latency_ms: Optional[float] = None
    """Wall-clock latency of the operation that produced this event."""

    token_usage: Optional[Dict[str, int]] = None
    """Flat ``input/output/total`` token counts when applicable."""

    payload: Dict[str, Any] = Field(default_factory=dict)
    """Event-specific data (tool name/args, stdout snippet, error message, ...)."""


class AgentTrace(BaseModel):
    """Complete agent trajectory attached to a sample."""

    strategy: Optional[str] = None
    """Registered strategy name used for this run."""

    environment: Optional[str] = None
    """Registered environment name used for this run, if any."""

    max_steps: int = 0
    """Configured ``max_steps`` cap."""

    events: List[AgentTraceEvent] = Field(default_factory=list)
    """Ordered event log."""

    def add(self, event: AgentTraceEvent) -> None:
        self.events.append(event)

    def add_event(
        self,
        *,
        step: int,
        type: EventType,
        message_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        token_usage: Optional[Dict[str, int]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AgentTraceEvent:
        """Convenience factory used by the loop implementation."""
        event = AgentTraceEvent(
            step=step,
            type=type,
            message_id=message_id,
            latency_ms=latency_ms,
            token_usage=token_usage,
            payload=payload or {},
        )
        self.events.append(event)
        return event

    @property
    def step_count(self) -> int:
        """Number of distinct ``step`` indices observed in the event log."""
        return len({ev.step for ev in self.events})


__all__ = ['AgentTrace', 'AgentTraceEvent', 'EventType']
