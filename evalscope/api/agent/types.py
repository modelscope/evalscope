"""Core data types for the Agent Loop evaluation framework.

These types form the contract between the AgentLoop, Strategy,
Environment and ToolExecutor.  They are intentionally lightweight
(pydantic / dataclasses) so that a strategy author only needs to
import from ``evalscope.api.agent`` to participate.
"""

from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from evalscope.api.messages import ChatMessage, Content
from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolCall, ToolInfo
from .mcp.types import MCPServerConfig

if TYPE_CHECKING:
    from .trace import AgentTrace


class BaseAgentConfig(BaseModel):
    """Fields shared by every agent-config variant.

    Both :class:`NativeAgentConfig` (the AgentLoop path) and
    :class:`ExternalAgentConfig` (the external-CLI bridge path) need an
    Agent execution environment plus a free-form ``kwargs`` channel;
    lifting them here avoids duplicating the schema across both subclasses.
    """

    # ``extra='forbid'`` so legacy ``extra=...`` configs (renamed to
    # ``kwargs``) and field typos surface as ValidationError instead of
    # silently being dropped — agent_config is a public surface.
    model_config = ConfigDict(extra='forbid')

    environment: Optional[str] = Field(default=None)
    """Registered Agent execution environment. ``None`` means no environment."""

    environment_extra: Dict[str, Any] = Field(default_factory=dict)
    """Options passed to the Agent execution environment constructor."""

    kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Free-form variant-specific options.  Native: forwarded to the
    strategy constructor.  External: forwarded to the runner constructor."""

    skills_dir: Optional[str] = Field(default=None)
    """Optional Agent Skills directory. Only agent runners consume this field."""

    skill_prompt_nudge: bool = Field(default=True)
    """Whether to add a neutral prompt nudge when skills are available."""


class NativeAgentConfig(BaseAgentConfig):
    """AgentLoop-driven agent configuration.

    When carried by ``TaskConfig.agent_config``, every
    ``DefaultDataAdapter``-based benchmark routes inference through the
    AgentLoop instead of calling ``model.generate`` once.  AgentLoopAdapter
    subclasses keep their benchmark defaults when fields are omitted and
    accept explicitly configured strategy, tools and max_steps values.
    """

    mode: Literal['native'] = Field(default='native')
    """Union discriminator — fixed value for the native AgentLoop path."""

    strategy: str = Field(default='function_calling')
    """Registered strategy name (``function_calling`` / ``react`` / ...)."""

    tools: List[str] = Field(default_factory=list)
    """Whitelist of registered tool names.  Empty = no tools (pure multi-turn)."""

    max_steps: int = Field(default=10)
    """Hard upper bound on loop iterations."""

    command_timeout: Optional[float] = Field(default=None)
    """Default timeout in seconds for native command-style tools.

    Tool calls that explicitly pass a timeout keep their own value. ``None``
    means use each tool's built-in default.
    """

    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)
    """List of MCP servers spawned alongside this agent's per-sample loop.

    Each entry runs a host-side MCP server (stdio or HTTP) and merges its
    advertised tools into the loop's tool set, alongside any benchmark-
    native tools.  The ``mcp`` Python SDK is imported lazily inside
    :class:`MCPServer.__aenter__`, so configurations with empty
    ``mcp_servers`` (the default) do not require ``pip install mcp``.
    """

    @field_validator('max_steps')
    @classmethod
    def _validate_max_steps(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('max_steps must be greater than 0.')
        return v

    @field_validator('command_timeout')
    @classmethod
    def _validate_command_timeout(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError('command_timeout must be greater than 0.')
        return v


class ExecResult(BaseModel):
    """Result of executing a command in an ``AgentEnvironment``."""

    returncode: int = 0
    stdout: str = ''
    stderr: str = ''
    timed_out: bool = False
    duration: float = 0.0
    extra: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class ToolExecutionOutput:
    """Rich observation returned by tools that produce attachments or termination."""

    text: str
    attachments: List[Content] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    terminate: bool = False
    final_answer: Optional[str] = None


class TurnOutcome(str, Enum):
    """What a parsed turn asks the loop to do next.

    The loop must not re-derive this from ``ParsedAction``'s field shape:
    ``tool_calls`` being empty is a *symptom* shared by two very different
    states (a malformed turn that tried to act, and an idle turn that only
    produced text), and conflating them is what made one nudge budget, one
    terminal exit and one trace label serve both.

    ``done`` is deliberately absent: termination is decided by the strategy
    hook ``AgentStrategy.is_done``, which is overridable, so folding it into a
    derived property would replace strategy semantics with a fixed rule.
    """

    ACT = 'act'
    """Has tool calls; the loop should execute them."""

    MALFORMED = 'malformed'
    """Tried to act but violated the protocol; ``error`` describes the rule."""

    IDLE = 'idle'
    """Only text — either the model's answer, or a stall."""


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

    @property
    def outcome(self) -> TurnOutcome:
        """Classify this turn for the loop; see :class:`TurnOutcome`.

        This is the single place the precedence is defined. ``ACT`` wins over
        ``MALFORMED`` so a strategy that reports usable tool calls *and* a
        complaint (e.g. "I dropped the calls past the cap") still makes
        progress; the complaint reaches the trace through the loop's parse
        error event.
        """
        if self.tool_calls:
            return TurnOutcome.ACT
        if self.error:
            return TurnOutcome.MALFORMED
        return TurnOutcome.IDLE


@dataclass
class AgentContext:
    """Mutable context shared across AgentLoop iterations.

    Strategies and tool executors read/write this object; the loop itself
    only bumps ``step`` / the nudge counters and appends messages.  Not
    serialized: persistence happens through ``AgentTrace``.
    """

    sample_id: Any
    messages: List[ChatMessage]
    tools: List[ToolInfo] = field(default_factory=list)
    step: int = 0
    max_steps: int = 10
    last_output: Optional[ModelOutput] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    nudge_count: int = 0
    """Idle nudges injected since the last turn that produced tool calls.

    Owned by :class:`AgentLoop`: it increments the counter when it injects a
    reminder and resets it once the model calls a tool again. Strategies read
    it in ``should_nudge`` and must not mutate it. Deriving the count by
    matching reminder text in ``messages`` instead would silently return 0
    whenever the reminder wording or shape changes, which reads as "never
    nudged" and lets a stuck model burn the whole step budget.
    """

    parse_error_nudge_count: int = 0
    """Malformed-turn nudges since the last turn that produced tool calls.

    Counted separately from :attr:`nudge_count` because the two failures are
    not interchangeable: a malformed turn is told the exact rule it broke and
    is therefore far more recoverable than a model that keeps answering in
    prose. Sharing one counter let either failure consume the other's retries.
    Also owned by :class:`AgentLoop`; strategies must not mutate it.
    """


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


__all__ = [
    'AgentContext',
    'AgentLoopResult',
    'BaseAgentConfig',
    'ExecResult',
    'NativeAgentConfig',
    'ParsedAction',
    'ToolSchemaMode',
    'ToolExecutionOutput',
    'TurnOutcome',
]
