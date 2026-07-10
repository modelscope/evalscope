from __future__ import annotations

import asyncio
from pydantic import ValidationError
from typing import Any, Mapping, Union

from evalscope.perf.config import (
    BenchmarkSuite,
    ClosedLoopLoad,
    ConversationLoad,
    GenerationConfig,
    MetricsConfig,
    OpenLoopLoad,
    OutputConfig,
    PerfConfig,
    RuntimeConfig,
    SLAConfig,
    TargetConfig,
    WarmupConfig,
    WorkloadConfig,
)
from evalscope.perf.domain.errors import PerfConfigError, PerfError, PerfUsageError
from evalscope.perf.domain.result import PerfSuiteResult
from evalscope.perf.engine import SuiteRunner


async def async_run_perf(config: Union[PerfConfig, Mapping[str, Any]]) -> PerfSuiteResult:
    """Run a performance benchmark suite asynchronously."""
    try:
        resolved = PerfConfig.from_input(config)
    except ValidationError as e:
        raise PerfConfigError(str(e)) from e
    return await SuiteRunner(resolved).run()


def run_perf(config: Union[PerfConfig, Mapping[str, Any]]) -> PerfSuiteResult:
    """Run a performance benchmark suite from synchronous code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_run_perf(config))
    raise PerfUsageError('run_perf() cannot be called from an active event loop; use await async_run_perf()')


__all__ = [
    'BenchmarkSuite',
    'ClosedLoopLoad',
    'ConversationLoad',
    'GenerationConfig',
    'MetricsConfig',
    'OpenLoopLoad',
    'OutputConfig',
    'PerfConfig',
    'PerfError',
    'PerfSuiteResult',
    'RuntimeConfig',
    'SLAConfig',
    'TargetConfig',
    'WarmupConfig',
    'WorkloadConfig',
    'async_run_perf',
    'run_perf',
]
