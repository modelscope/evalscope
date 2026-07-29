"""Stateful task-environment session contract."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from .types import EnvironmentStepResult


class TaskEnvironmentSession(ABC):
    """One stateful task episode exposed through reset/step/state."""

    backend_name: str = 'base'

    @abstractmethod
    async def reset(self, **kwargs: Any) -> EnvironmentStepResult:
        """Reset the episode and return its initial observation."""
        ...

    @abstractmethod
    async def step(self, action: Dict[str, Any]) -> EnvironmentStepResult:
        """Apply one action to the episode."""
        ...

    @abstractmethod
    async def state(self) -> Dict[str, Any]:
        """Return service state used for capability validation."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the session connection."""
        ...


__all__ = ['TaskEnvironmentSession']
