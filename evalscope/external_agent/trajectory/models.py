"""ATIF-style trajectory data model.

Captures one external-agent run as an ordered list of :class:`Step` entries
derived from the LLM request/response stream observed at the bridge.  The
shape is deliberately flat and JSON-friendly so it round-trips through
``sample.metadata`` and the prediction cache without custom serialisers.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


class ToolCallRecord(BaseModel):
    """One tool invocation requested by the agent in an assistant turn."""

    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """One tool result fed back to the agent in a subsequent user turn."""

    tool_call_id: Optional[str] = Field(default=None)
    tool_name: Optional[str] = Field(default=None)
    output: str = ''
    is_error: bool = False


class TurnUsage(BaseModel):
    """Token usage for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


StepSource = Literal['agent', 'tool', 'user', 'system']


class Step(BaseModel):
    """One element in the unified trajectory.

    A single LLM call typically yields one ``agent`` step (with optional
    tool calls) plus zero or more ``tool`` steps materialised from the
    ``tool_result`` blocks present in the next request.
    """

    step_id: int
    source: StepSource
    message: Optional[str] = Field(default=None)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    usage: Optional[TurnUsage] = Field(default=None)


class Trajectory(BaseModel):
    """Complete trace of one external-agent run."""

    framework: str
    model: Optional[str] = Field(default=None)
    trial_id: Optional[str] = Field(default=None)
    steps: List[Step] = Field(default_factory=list)
    total_usage: TurnUsage = Field(default_factory=TurnUsage)
    metadata: Dict[str, Any] = Field(default_factory=dict)
