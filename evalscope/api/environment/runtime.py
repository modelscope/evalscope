"""Runtime contract for hosting task-environment services."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class EnvironmentRuntimeLease(ABC):
    """Endpoint and lifecycle for one hosted task-environment service."""

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
    ) -> EnvironmentRuntimeLease:
        """Start or attach to one task-environment service."""
        ...


__all__ = ['EnvironmentRuntime', 'EnvironmentRuntimeLease']
