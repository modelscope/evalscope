from __future__ import annotations

import asyncio
from typing import AsyncIterator, List, Optional

from evalscope.perf.config.models import OpenLoopLoad
from evalscope.perf.domain.errors import PerfRunError
from evalscope.perf.domain.observation import RequestObservation
from evalscope.perf.domain.workload import SingleTurnItem
from evalscope.perf.engine.schedulers.base import Scheduler


class OpenLoopScheduler(Scheduler):

    async def run(self) -> None:
        load = self.context.spec.load
        assert isinstance(load, OpenLoopLoad)
        if self.context.spec.warmup_count:
            await self._warmup(self.context.spec.warmup_count, load.max_outstanding)
        await self._benchmark(load)

    async def _warmup(self, count: int, max_outstanding: int) -> None:
        pending: set[asyncio.Task] = set()
        for _ in range(count):
            item = await self.cursor.next()
            if item is None:
                break
            if not isinstance(item, SingleTurnItem):
                raise TypeError('Open-loop scheduler requires single-turn work items')
            if len(pending) >= max_outstanding:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    await self.context.emit(task.result())
            pending.add(asyncio.create_task(self.context.execute(item, is_warmup=True)))
        if pending:
            for observation in await asyncio.gather(*pending):
                await self.context.emit(observation)

    async def _intervals(self, load: OpenLoopLoad) -> AsyncIterator[float]:
        if load.arrival == 'constant':
            while True:
                yield 1.0 / load.request_rate
        if load.arrival == 'poisson':
            while True:
                yield self.context.rng.expovariate(load.request_rate)
        batch_size = 128
        while True:
            values = [self.context.rng.expovariate(load.request_rate) for _ in range(batch_size)]
            scale = (batch_size / load.request_rate) / sum(values)
            for value in values:
                yield value * scale

    async def _benchmark(self, load: OpenLoopLoad) -> None:
        pending: set[asyncio.Task] = set()
        start = self.context.clock()
        deadline = start + load.duration if load.duration is not None else None
        scheduled = start
        dispatched = 0
        intervals = self._intervals(load)

        try:
            while load.request_count is None or dispatched < load.request_count:
                interval = await intervals.__anext__()
                scheduled += interval
                if deadline is not None and scheduled >= deadline:
                    break
                sleep_for = scheduled - self.context.clock()
                if sleep_for > 0:
                    await self.context.sleep(sleep_for)
                done = {task for task in pending if task.done()}
                pending -= done
                for task in done:
                    await self.context.emit(task.result())
                item = await self.cursor.next()
                if item is None:
                    break
                if not isinstance(item, SingleTurnItem):
                    raise TypeError('Open-loop scheduler requires single-turn work items')
                dispatched += 1
                if len(pending) >= load.max_outstanding:
                    if load.overflow_policy == 'fail':
                        raise PerfRunError(self.context.run_id, 'open_loop', 'max_outstanding was exceeded')
                    await self.context.emit(
                        RequestObservation(
                            run_id=self.context.run_id,
                            request_id=self.context.next_request_id(),
                            scheduled_time=scheduled,
                            dispatch_time=self.context.clock(),
                            dropped=True,
                            drop_reason='max_outstanding',
                            outstanding=len(pending),
                        )
                    )
                    continue
                task = asyncio.create_task(
                    self.context.execute(
                        item,
                        is_warmup=False,
                        scheduled_time=scheduled,
                        outstanding=len(pending) + 1,
                    )
                )
                pending.add(task)
            if pending:
                for observation in await asyncio.gather(*pending):
                    await self.context.emit(observation)
        except BaseException:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
