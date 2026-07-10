from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from evalscope.perf.domain.workload import WorkItem
from evalscope.perf.engine.context import RunContext


class ItemCursor:
    """Serialize access to one async workload iterator across workers."""

    def __init__(self, iterator: AsyncIterator[WorkItem]) -> None:
        self._iterator = iterator
        self._lock = asyncio.Lock()

    async def next(self) -> Optional[WorkItem]:
        async with self._lock:
            try:
                return await self._iterator.__anext__()
            except StopAsyncIteration:
                return None


class Scheduler(ABC):

    def __init__(self, context: RunContext, items: AsyncIterator[WorkItem]) -> None:
        self.context = context
        self.cursor = ItemCursor(items)

    @abstractmethod
    async def run(self) -> None:
        raise NotImplementedError
