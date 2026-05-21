"""External agent integration (HTTP bridge + runner).

Drives third-party agent CLIs (claude-code, codex, openhands, ...) inside a
sandbox while routing their LLM calls through an in-process reverse proxy
that forwards to EvalScope's model layer.  The bridge captures the full
LLM request/response stream into the shared :class:`AgentTrace` format
(same one populated by :class:`AgentLoop` for native runs).
"""

from evalscope.api.registry import RUNNER_REGISTRY, get_runner, list_runners, register_runner
from . import runners as _runners  # noqa: F401  (side-effect: registers built-in runners)
from .config import BridgeConfig, ExternalAgentConfig, ExternalAgentFramework
from .runners.base import AgentRunner, AgentRunResult, BridgeEndpoint, ExternalAgentTask

__all__ = [
    'AgentRunResult',
    'AgentRunner',
    'BridgeConfig',
    'BridgeEndpoint',
    'ExternalAgentConfig',
    'ExternalAgentFramework',
    'ExternalAgentTask',
    'RUNNER_REGISTRY',
    'get_runner',
    'list_runners',
    'register_runner',
]
