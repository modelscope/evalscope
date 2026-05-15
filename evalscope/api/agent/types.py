"""Core data types for the Agent Loop evaluation framework.

These types form the contract between the AgentLoop, Strategy,
Environment and ToolExecutor.  They are intentionally lightweight
(pydantic / dataclasses) so that a strategy author only needs to
import from ``evalscope.api.agent`` to participate.
"""

from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from evalscope.api.messages import ChatMessage
from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolCall, ToolInfo

if TYPE_CHECKING:
    from .trace import AgentTrace


class AgentConfig(BaseModel):
    """Global agent configuration carried by ``TaskConfig.agent_config``.

    When set, every ``DefaultDataAdapter``-based benchmark will route
    inference through the AgentLoop instead of calling ``model.generate``
    once.  Individual AgentAdapter subclasses (e.g. SWE-bench_Pro) ignore
    this global config and use their own settings to avoid double wrapping.
    """

    strategy: str = Field(default='function_calling')
    """Registered strategy name (``function_calling`` / ``react`` / ...)."""

    tools: List[str] = Field(default_factory=list)
    """Whitelist of registered tool names.  Empty = no tools (pure multi-turn)."""

    max_steps: int = Field(default=10)
    """Hard upper bound on loop iterations."""

    environment: Optional[str] = Field(default=None)
    """Registered environment name.  ``None`` means no sandbox (local tools only)."""

    extra: Dict[str, Any] = Field(default_factory=dict)
    """Free-form strategy-specific options (passed to strategy constructor)."""

    environment_extra: Dict[str, Any] = Field(default_factory=dict)
    """Free-form environment-specific options (passed to environment constructor)."""


class ExecResult(BaseModel):
    """Result of executing a command in an ``AgentEnvironment``."""

    returncode: int = 0
    stdout: str = ''
    stderr: str = ''
    timed_out: bool = False
    duration: float = 0.0
    extra: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class ParsedAction:
    """Structured output produced by :class:`AgentStrategy.parse_output`.

    A strategy converts raw ``ModelOutput`` into one of three shapes:

    * ``tool_calls`` non-empty  → loop will execute them
    * ``final_answer`` not None → loop terminates, use this as prediction
    * ``error`` not None        → loop injects an error observation and retries
    """

    tool_calls: List[ToolCall] = field(default_factory=list)
    final_answer: Optional[str] = None
    error: Optional[str] = None
    raw_text: Optional[str] = None


@dataclass
class AgentContext:
    """Mutable context shared across AgentLoop iterations.

    Strategies and tool executors read/write this object; the loop itself
    only bumps ``step`` and appends messages.  Not serialized: persistence
    happens through ``AgentTrace`` attached to ``TaskState``.
    """

    sample_id: Any
    messages: List[ChatMessage]
    tools: List[ToolInfo] = field(default_factory=list)
    step: int = 0
    max_steps: int = 10
    last_output: Optional[ModelOutput] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


ToolSchemaMode = Literal['function_calling', 'textual_block', 'none']
"""How a strategy surfaces tool definitions to the model.

* ``function_calling``: pass ``ToolInfo`` list to ``model.generate``
* ``textual_block``:    render tool specs into system prompt (ReAct style)
* ``none``:             no tool surface at all (pure thinking loop)
"""


@dataclass
class AgentLoopResult:
    """Aggregate returned by :meth:`AgentLoop.run`."""

    messages: List[ChatMessage]
    final_output: ModelOutput
    trace: 'AgentTrace'
    final_submission: Optional[str] = None
    """Sentinel-detected submission payload, set when a strategy raises
    :class:`evalscope.api.agent.exceptions.Submitted` during the loop.

    Strategies that rely on the sentinel protocol (e.g. ``swe_bench_toolcall``)
    read this in :meth:`extract_final_answer` instead of scanning messages.
    """


__all__ = [
    'AgentConfig',
    'AgentContext',
    'AgentLoopResult',
    'ExecResult',
    'ParsedAction',
    'ToolSchemaMode',
]
