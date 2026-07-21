import asyncio
import pytest
from types import SimpleNamespace
from typing import AsyncIterator, Dict, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies import open_loop
from evalscope.perf.core.strategies.open_loop import OpenLoopStrategy
from evalscope.perf.utils.body_meta import BODY_META_ARRIVAL_OFFSET


class FakeClient:

    def __init__(self) -> None:
        self.request_ids: List[int] = []

    async def post(self, request: Dict[str, int]) -> SimpleNamespace:
        self.request_ids.append(request['id'])
        return SimpleNamespace(is_warmup=False)


@pytest.mark.parametrize('use_trace_offsets', [True, False], ids=['trace', 'poisson'])
def test_schedule_stops_dispatching_at_duration_deadline(
    monkeypatch: pytest.MonkeyPatch,
    use_trace_offsets: bool,
) -> None:
    current_time = 0.0

    def perf_counter() -> float:
        return current_time

    async def sleep(seconds: float) -> None:
        nonlocal current_time
        current_time += seconds

    async def request_generator(requests: List[dict]) -> AsyncIterator[Tuple[dict, bool]]:
        for request in requests:
            yield request, False

    async def run() -> List[int]:
        requests = [{'id': 1}, {'id': 2}]
        if use_trace_offsets:
            requests[0][BODY_META_ARRIVAL_OFFSET] = 0.0
            requests[1][BODY_META_ARRIVAL_OFFSET] = 1.0
        else:
            monkeypatch.setattr(
                open_loop.np.random, 'exponential', lambda *args, **kwargs: open_loop.np.array([0.0, 1.0])
            )
        args = Arguments(model='m', api='openai', open_loop=True, rate=1.0, number=2, duration=0.5)
        args.rate = 1.0
        args.number = 2
        client = FakeClient()
        strategy = OpenLoopStrategy(args, None, client, asyncio.Queue(), request_generator(requests))
        await strategy.run()
        return client.request_ids

    monkeypatch.setattr(open_loop.time, 'perf_counter', perf_counter)
    monkeypatch.setattr(open_loop.asyncio, 'sleep', sleep)

    assert asyncio.run(run()) == [1]
