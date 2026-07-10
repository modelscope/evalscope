import asyncio
import time
from typing import AsyncIterator, Iterator

from evalscope.perf import BenchmarkSuite, OpenLoopLoad, PerfConfig, TargetConfig
from evalscope.perf.config.resolve import ResolvedRunSpec
from evalscope.perf.domain.observation import RequestObservation, TransportEvent
from evalscope.perf.domain.workload import SingleTurnItem, WorkItem
from evalscope.perf.engine.context import RunContext
from evalscope.perf.engine.run_engine import RunEngine
from evalscope.perf.engine.schedulers.open_loop import OpenLoopScheduler
from evalscope.perf.protocols.openai_chat import OpenAIChatProtocol
from evalscope.perf.results.store import ResultStore
from evalscope.perf.transport.base import HttpRequest, HttpTransport


class MemoryStore(ResultStore):

    def __init__(self) -> None:
        self.items = []

    def open(self) -> None:
        pass

    def write(self, observation: RequestObservation) -> None:
        self.items.append(observation)

    def observations(self, include_warmup: bool = False) -> Iterator[RequestObservation]:
        return iter(self.items)

    def close(self) -> None:
        pass


class BoundedFakeTransport(HttpTransport):

    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass

    async def events(self, request: HttpRequest) -> AsyncIterator[TransportEvent]:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        now = time.perf_counter()
        try:
            yield TransportEvent(kind='request_start', timestamp=now)
            yield TransportEvent(kind='response_start', timestamp=now, status_code=200)
            await asyncio.sleep(0)
            yield TransportEvent(
                kind='json',
                timestamp=time.perf_counter(),
                data={
                    'choices': [{
                        'message': {
                            'content': 'ok'
                        }
                    }],
                    'usage': {
                        'prompt_tokens': 1,
                        'completion_tokens': 1
                    },
                },
            )
            yield TransportEvent(kind='response_end', timestamp=time.perf_counter())
        finally:
            self.active -= 1


async def _items() -> AsyncIterator[WorkItem]:
    while True:
        yield SingleTurnItem(messages='hello')


def test_large_open_loop_keeps_request_tasks_bounded() -> None:

    async def run() -> tuple[BoundedFakeTransport, MemoryStore]:
        load = OpenLoopLoad(
            request_rate=1_000_000,
            request_count=5000,
            max_outstanding=3,
            arrival='constant',
        )
        config = PerfConfig(target=TargetConfig(model='fake'), suite=BenchmarkSuite(loads=[load]))
        spec = ResolvedRunSpec(load_id='load', load=load, seed=1, warmup_count=0, item_limit=5000)
        transport = BoundedFakeTransport()
        store = MemoryStore()
        context = RunContext(
            run_id='run',
            config=config,
            spec=spec,
            transport=transport,
            protocol=OpenAIChatProtocol(config.target),
            store=store,
            queue=asyncio.Queue(maxsize=4),
            sleep=lambda _: asyncio.sleep(0),
        )
        consumer = asyncio.create_task(context.consume())
        await OpenLoopScheduler(context, _items()).run()
        await context.finish()
        await consumer
        return transport, store

    transport, store = asyncio.run(run())
    assert transport.maximum_active <= 3
    assert len(store.items) == 5000
    assert max(item.outstanding or 0 for item in store.items) <= 3


def test_scheduler_error_flushes_observations_and_leaves_no_tasks(tmp_path) -> None:

    async def run() -> tuple[MemoryStore, int]:
        load = OpenLoopLoad(request_rate=1, request_count=1, max_outstanding=1)
        config = PerfConfig(target=TargetConfig(model='fake'), suite=BenchmarkSuite(loads=[load]))
        spec = ResolvedRunSpec(load_id='load', load=load, seed=1, warmup_count=0, item_limit=1)
        store = MemoryStore()
        context = RunContext(
            run_id='run',
            config=config,
            spec=spec,
            transport=BoundedFakeTransport(),
            protocol=OpenAIChatProtocol(config.target),
            store=store,
        )

        class FailingScheduler:

            async def run(self) -> None:
                await context.emit(RequestObservation(run_id='run', request_id='completed', success=True))
                raise RuntimeError('worker failed')

        engine = RunEngine(config, 'run', spec, str(tmp_path))
        try:
            await engine._run_scheduler(context, FailingScheduler())
        except RuntimeError as e:
            assert str(e) == 'worker failed'
        await asyncio.sleep(0)
        remaining = [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
        return store, len(remaining)

    store, remaining = asyncio.run(run())
    assert [item.request_id for item in store.items] == ['completed']
    assert remaining == 0
