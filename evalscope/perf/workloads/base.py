from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from evalscope.perf.config.models import PerfConfig
from evalscope.perf.config.resolve import ResolvedRunSpec
from evalscope.perf.domain.workload import WorkItem, WorkloadMeta


@dataclass(frozen=True)
class WorkloadContext:
    config: PerfConfig


class WorkloadSource(ABC):
    """Base class for lazy, typed workload sources."""

    meta: WorkloadMeta

    def __init__(self, context: WorkloadContext) -> None:
        self.context = context

    async def prepare(self) -> None:
        """Validate and initialize the source before dispatch begins."""

    @abstractmethod
    async def iter_items(self, run: ResolvedRunSpec) -> AsyncIterator[WorkItem]:
        """Yield workload items lazily for one run."""
        if False:
            yield  # pragma: no cover
