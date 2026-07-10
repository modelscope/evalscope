from __future__ import annotations

import asyncio
from typing import Optional

from evalscope.perf.config.models import ClosedLoopLoad
from evalscope.perf.domain.workload import SingleTurnItem
from evalscope.perf.engine.schedulers.base import Scheduler


class ClosedLoopScheduler(Scheduler):

    async def run(self) -> None:
        load = self.context.spec.load
        assert isinstance(load, ClosedLoopLoad)
        if self.context.spec.warmup_count:
            await self._phase(self.context.spec.warmup_count, True, None, load.concurrency)
        deadline = self.context.clock() + load.duration if load.duration is not None else None
        await self._phase(load.request_count, False, deadline, load.concurrency)

    async def _phase(
        self,
        limit: Optional[int],
        is_warmup: bool,
        deadline: Optional[float],
        concurrency: int,
    ) -> None:
        claimed = 0
        claim_lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal claimed
            while not self.context.cancelled.is_set():
                async with claim_lock:
                    if deadline is not None and self.context.clock() >= deadline:
                        return
                    if limit is not None and claimed >= limit:
                        return
                    claimed += 1
                item = await self.cursor.next()
                if item is None:
                    return
                if not isinstance(item, SingleTurnItem):
                    raise TypeError('Closed-loop scheduler requires single-turn work items')
                observation = await self.context.execute(item, is_warmup=is_warmup)
                await self.context.emit(observation)

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
