"""Sample-scoped runtime capabilities used by agent tools and CLI runners."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from .types import ExecResult


class AgentRuntime(ABC):
    """Per-sample command runtime.

    Runtimes provide shell and optional file-transfer capabilities to agent
    tools and external CLI runners. Stateful task environments are represented
    separately by :class:`evalscope.api.environment.TaskEnvironmentSession`.
    """

    name: str = 'base'
    """Registered runtime name. Subclasses should override."""

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        """Run a command inside the runtime and return its result.

        ``env`` (when provided) is merged on top of the environment's default
        variables before the command runs.  Subclasses that cannot honour
        ``env`` should raise ``NotImplementedError`` rather than silently
        dropping the request.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support exec')

    @abstractmethod
    async def close(self) -> None:
        """Release any external resources (containers, temp dirs, ...)."""
        ...

    async def put_dir(self, source_dir: str | Path, target_dir: str) -> None:
        """Copy a host directory into the environment.

        Environments that do not expose a writable filesystem bridge should
        raise ``NotImplementedError``. The default keeps older custom
        environments source-compatible while allowing runners to probe for the
        capability when they need host-side assets such as Agent Skills.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support put_dir')

    async def __aenter__(self) -> 'AgentRuntime':
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


__all__ = ['AgentRuntime']
