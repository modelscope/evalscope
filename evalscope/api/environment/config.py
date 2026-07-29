"""Typed configuration for stateful task environments."""

from pydantic import ConfigDict, Field
from typing import Any, Dict

from evalscope.utils.argument_utils import BaseArgument


class EnvironmentRuntimeConfig(BaseArgument):
    """Configuration for the runtime hosting a task-environment service."""

    model_config = ConfigDict(extra='forbid')

    name: str
    config: Dict[str, Any] = Field(default_factory=dict)


class TaskEnvironmentConfig(BaseArgument):
    """Configuration for a task protocol backend and its hosting runtime."""

    model_config = ConfigDict(extra='forbid')

    backend: str
    backend_args: Dict[str, Any] = Field(default_factory=dict)
    runtime: EnvironmentRuntimeConfig


__all__ = ['EnvironmentRuntimeConfig', 'TaskEnvironmentConfig']
