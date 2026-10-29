"""Demand-load the bounded tools and rubrics of the pinned public benchmark."""

from importlib import import_module
from typing import Awaitable, Callable

from evalscope.api.tool import ToolInfo
from evalscope.api.tool.tool_info import parse_tool_info
from evalscope.benchmarks.agentharm.types import AgentHarmSampleMeta, Rubric
from evalscope.benchmarks.agentharm.upstream.resources import MANIFEST


def tool_function(task_kind: str, name: str) -> Callable[..., Awaitable[str]]:
    """Build a fresh ordinary async function for one official mock tool."""
    if task_kind not in ('harmful', 'benign') or name not in MANIFEST[task_kind]['tools']:
        raise ValueError(f'Unknown AgentHarm {task_kind} tool: {name!r}')
    namespace = 'common_tools' if name in MANIFEST['common_tools'] else f'{task_kind}_tools'
    module = import_module(f'evalscope.benchmarks.agentharm.upstream.{namespace}.{name}')
    return getattr(module, name)()


def tool_info(task_kind: str, name: str) -> ToolInfo:
    """Use the shared schema builder while retaining the official factory name."""
    info = parse_tool_info(tool_function(task_kind, name))
    info.name = name
    return info


def rubric(meta: AgentHarmSampleMeta, messages: list) -> Rubric:
    """Start one task's ordinary Python grading generator, without any model I/O."""
    if meta.grading_function not in MANIFEST[meta.task_kind]['rubrics']:
        raise ValueError(f'Unknown AgentHarm rubric: {meta.grading_function!r}')
    module = import_module(f'evalscope.benchmarks.agentharm.upstream.{meta.task_kind}_grading')
    return getattr(module, meta.grading_function)(meta, messages)
