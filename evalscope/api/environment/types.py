"""Data contracts shared by task-environment backends and benchmarks."""

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, Optional


class EnvironmentStepResult(BaseModel):
    """Environment-agnostic result returned by reset and step operations."""

    model_config = ConfigDict(extra='forbid')

    observation: Dict[str, Any] = Field(default_factory=dict)
    reward: Optional[float] = None
    done: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = ['EnvironmentStepResult']
