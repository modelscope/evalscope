"""Runtime contract for hosting task-environment services."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class EnvironmentRuntimeHandle(ABC):
    """One running task-environment service owned by the caller.

    A runtime is a reusable service factory. Each ``start`` call returns a
    separate handle carrying that service's endpoint, logs and cleanup.
    """

    name: str = 'base'
    base_url: str

    async def capture_logs(self, destination: str | Path) -> bool:
        """Persist service logs when available."""
        return False

    @abstractmethod
    async def close(self) -> None:
        """Release the hosted service."""
        ...


class EnvironmentRuntime(ABC):
    """Start a task-environment service process."""

    @abstractmethod
    async def start(
        self,
        *,
        image: Optional[str],
        env_vars: Dict[str, str],
        config: Dict[str, Any],
    ) -> EnvironmentRuntimeHandle:
        """Start or attach to one task-environment service."""
        ...


__all__ = ['EnvironmentRuntime', 'EnvironmentRuntimeHandle']
