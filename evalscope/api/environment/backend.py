"""Task-environment protocol backend contract."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from .session import TaskEnvironmentSession


class TaskEnvironmentBackend(ABC):
    """Create protocol sessions against an already running service endpoint."""

    @abstractmethod
    def create_session(
        self,
        *,
        base_url: str,
        config: Dict[str, Any],
    ) -> TaskEnvironmentSession:
        """Create a lazily connected task-environment session."""
        ...


__all__ = ['TaskEnvironmentBackend']
