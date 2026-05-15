"""Public interfaces for the Agent evaluation framework.

Mirrors the ``evalscope/api/model`` + ``evalscope/models`` split:
this ``api`` package defines contracts only; concrete implementations
live in ``evalscope/agent``.
"""

from .environment import AgentEnvironment
from .exceptions import Submitted
from .loop import AgentLoop
from .strategy import AgentStrategy
from .tool_executor import ToolExecutor, ToolHandler
from .trace import AgentTrace, AgentTraceEvent, EventType
from .types import AgentConfig, AgentContext, AgentLoopResult, ExecResult, ParsedAction, ToolSchemaMode

__all__ = [
    'AgentConfig',
    'AgentContext',
    'AgentEnvironment',
    'AgentLoop',
    'AgentLoopResult',
    'AgentStrategy',
    'AgentTrace',
    'AgentTraceEvent',
    'EventType',
    'ExecResult',
    'ParsedAction',
    'Submitted',
    'ToolExecutor',
    'ToolHandler',
    'ToolSchemaMode',
]
