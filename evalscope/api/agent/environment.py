"""Sample-scoped sandboxed execution environment.

``AgentEnvironment`` is deliberately minimal: it exposes a file-system +
exec surface and knows nothing about agents, strategies or tools.
Concrete implementations (EnclaveAgentEnvironment, LocalAgentEnvironment)
live in ``evalscope/agent/environments/``.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .types import ExecResult


class AgentEnvironment(ABC):
    """Per-sample isolated execution environment.

    Lifecycle: ``__aenter__`` → N × ``exec`` → ``close``.  Created and
    destroyed per sample by the AgentAdapter.
    """

    name: str = 'base'
    """Registered environment name.  Subclasses should override."""

    @abstractmethod
    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        """Run a command inside the environment and return its result.

        ``env`` (when provided) is merged on top of the environment's default
        variables before the command runs.  Subclasses that cannot honour
        ``env`` should raise ``NotImplementedError`` rather than silently
        dropping the request.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any external resources (containers, temp dirs, ...)."""
        ...

    async def __aenter__(self) -> 'AgentEnvironment':
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


__all__ = ['AgentEnvironment']
