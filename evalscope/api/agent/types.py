"""Core data types for the Agent Loop evaluation framework.

These types form the contract between the AgentLoop, Strategy,
Environment and ToolExecutor.  They are intentionally lightweight
(pydantic / dataclasses) so that a strategy author only needs to
import from ``evalscope.api.agent`` to participate.
"""

from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from evalscope.api.messages import ChatMessage
from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolCall, ToolInfo
from .mcp.types import MCPServerConfig

if TYPE_CHECKING:
    from .trace import AgentTrace


class BaseAgentConfig(BaseModel):
    """Fields shared by every agent-config variant.

    Both :class:`NativeAgentConfig` (the AgentLoop path) and
    :class:`ExternalAgentConfig` (the external-CLI bridge path) need an
    :class:`AgentEnvironment` plus a free-form ``kwargs`` channel; lifting
    them here avoids duplicating the schema across both subclasses.
    """

    # ``extra='forbid'`` so legacy ``extra=...`` configs (renamed to
    # ``kwargs``) and field typos surface as ValidationError instead of
    # silently being dropped — agent_config is a public surface.
    model_config = ConfigDict(extra='forbid')

    environment: Optional[str] = Field(default=None)
    """Registered environment name.  ``None`` means no sandbox (local tools only)."""

    environment_extra: Dict[str, Any] = Field(default_factory=dict)
    """Free-form environment-specific options (passed to environment constructor)."""

    kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Free-form variant-specific options.  Native: forwarded to the
    strategy constructor.  External: forwarded to the runner constructor."""


class NativeAgentConfig(BaseAgentConfig):
    """AgentLoop-driven agent configuration.

    When carried by ``TaskConfig.agent_config``, every
    ``DefaultDataAdapter``-based benchmark routes inference through the
    AgentLoop instead of calling ``model.generate`` once.  Individual
    AgentAdapter subclasses (e.g. SWE-bench_Pro) ignore this global config
    and use their own settings to avoid double wrapping.
    """

    mode: Literal['native'] = Field(default='native')
    """Union discriminator — fixed value for the native AgentLoop path."""

    strategy: str = Field(default='function_calling')
    """Registered strategy name (``function_calling`` / ``react`` / ...)."""

    tools: List[str] = Field(default_factory=list)
    """Whitelist of registered tool names.  Empty = no tools (pure multi-turn)."""

    max_steps: int = Field(default=10)
    """Hard upper bound on loop iterations."""

    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)
    """List of MCP servers spawned alongside this agent's per-sample loop.

    Each entry runs a host-side MCP server (stdio or HTTP) and merges its
    advertised tools into the loop's tool set, alongside any benchmark-
    native tools.  The ``mcp`` Python SDK is imported lazily inside
    :class:`MCPServer.__aenter__`, so configurations with empty
    ``mcp_servers`` (the default) do not require ``pip install mcp``.
    """


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


__all__ = [
    'AgentContext',
    'AgentLoopResult',
    'BaseAgentConfig',
    'ExecResult',
    'NativeAgentConfig',
    'ParsedAction',
    'ToolSchemaMode',
]
