"""External-agent runner abstractions and built-in implementations.

Registry helpers (``register_runner`` / ``get_runner`` / ...) live in
:mod:`evalscope.api.registry`; this subpackage only exposes runner
classes plus their value types.
"""

from evalscope.api.registry import get_runner
from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask, RunnerTimeoutError
from .claude_code import ClaudeCodeRunner  # noqa: F401  (side-effect: registers 'claude-code')
from .codex import CodexRunner  # noqa: F401  (side-effect: registers 'codex')
from .gemini_cli import GeminiCliRunner  # noqa: F401  (side-effect: registers 'gemini-cli')
from .mock import MockAgentRunner  # noqa: F401  (side-effect: registers 'mock')
from .opencode import OpenCodeRunner  # noqa: F401  (side-effect: registers 'opencode')

__all__ = [
    'AgentRunResult',
    'AgentRunner',
    'BridgeEndpoint',
    'ClaudeCodeRunner',
    'CodexRunner',
    'ExternalAgentTask',
    'GeminiCliRunner',
    'MockAgentRunner',
    'OpenCodeRunner',
    'RunnerTimeoutError',
    'get_runner',
]
