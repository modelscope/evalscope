"""Shared setup and controlled clients for closed-loop scheduling tests."""

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies.closed_loop import ClosedLoopStrategy


def _make_args(**kwargs: Any) -> Arguments:
    """Build Arguments with scalar (already swept) number/parallel/rate."""
    number = kwargs.pop('number', 6)
    parallel = kwargs.pop('parallel', 3)
    rate = kwargs.pop('rate', -1)
    args = Arguments(model='test-model', api='openai', number=number, parallel=parallel, rate=rate, **kwargs)
    args.number = number
    args.parallel = parallel
    args.rate = rate
    return args


class _RecordingClient:
    """Record server occupancy with deterministic, staggered completions."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.events: List[Dict[str, Any]] = []
        self._seq = 0

    async def post(self, request: Dict[str, Any]) -> SimpleNamespace:
        seq = self._seq
        self._seq += 1
        self.events.append({
            'request': request,
            'occupancy_at_start': self.in_flight,
            'order': seq,
        })
        self.in_flight += 1
        try:
            # Distinct event-loop tick counts free the slots one at a time.
            for _ in range(2 * (seq + 1)):
                await asyncio.sleep(0)
        finally:
            self.in_flight -= 1
        return SimpleNamespace(is_warmup=False)


def _run(args: Arguments, requests: List[Tuple[dict, bool]]) -> _RecordingClient:
    """Drive ClosedLoopStrategy over requests and return its recording client."""
    client = _RecordingClient()

    async def generator() -> AsyncIterator[Tuple[dict, bool]]:
        for item in requests:
            yield item

    async def main() -> None:
        queue: asyncio.Queue = asyncio.Queue()
        strategy = ClosedLoopStrategy(args, None, client, queue, generator())
        await strategy.run()

    asyncio.run(main())
    return client


def _stream(warmup: int, measured: int) -> List[Tuple[dict, bool]]:
    return [({'id': i}, i < warmup) for i in range(warmup + measured)]
