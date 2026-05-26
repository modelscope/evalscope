"""Public interfaces for the Agent evaluation framework.

Mirrors the ``evalscope/api/model`` + ``evalscope/models`` split:
this ``api`` package defines contracts only; concrete implementations
live in ``evalscope/agent``.
"""

from .environment import AgentEnvironment
from .loop import AgentLoop
from .strategy import AgentStrategy
from .tool_executor import ToolExecutor, ToolHandler
from .trace import AgentTrace, AgentTraceEvent, EventType
from .types import (
    AgentContext,
    AgentLoopResult,
    BaseAgentConfig,
    ExecResult,
    NativeAgentConfig,
    ParsedAction,
    ToolSchemaMode,
)

__all__ = [
    'AgentContext',
    'AgentEnvironment',
    'AgentLoop',
    'AgentLoopResult',
    'AgentStrategy',
    'AgentTrace',
    'AgentTraceEvent',
    'BaseAgentConfig',
    'EventType',
    'ExecResult',
    'NativeAgentConfig',
    'ParsedAction',
    'ToolExecutor',
    'ToolHandler',
    'ToolSchemaMode',
]
