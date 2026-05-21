"""External-agent runner abstractions and built-in implementations."""

from evalscope.api.registry import RUNNER_REGISTRY, get_runner, list_runners, register_runner
from .base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask
from .claude_code import ClaudeCodeRunner  # noqa: F401  (side-effect: registers 'claude-code')
from .mock import MockAgentRunner  # noqa: F401  (side-effect: registers 'mock')

__all__ = [
    'AgentRunResult',
    'AgentRunner',
    'BridgeEndpoint',
    'ClaudeCodeRunner',
    'ExternalAgentTask',
    'MockAgentRunner',
    'RUNNER_REGISTRY',
    'get_runner',
    'list_runners',
    'register_runner',
]
