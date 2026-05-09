import asyncio
import numpy as np
from typing import TYPE_CHECKING, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies.base import BenchmarkStrategy
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.perf.core.http_client import AioHttpClient
    from evalscope.perf.plugin.api.base import ApiPluginBase

logger = get_logger()


async def _send_request_open_loop(
    request: dict,
    is_warmup: bool,
    queue: asyncio.Queue,
    client: 'AioHttpClient',
) -> None:
    """Open-loop send: fires immediately regardless of in-flight count."""
    benchmark_data = await client.post(request)
    benchmark_data.is_warmup = is_warmup
    benchmark_data.update_gpu_usage()
    await queue.put(benchmark_data)


class OpenLoopStrategy(BenchmarkStrategy):
    """Open-loop benchmark strategy.

    Dispatches requests at the scheduled Poisson-arrival rate (``args.rate``)
    without a semaphore.  Requests are fired regardless of whether the server
    has finished processing previous ones.  This models realistic traffic
    patterns where arrivals are independent of service time.
    """

    def __init__(
        self,
        args: Arguments,
        api_plugin: 'ApiPluginBase',
        client: 'AioHttpClient',
        queue: asyncio.Queue,
        request_generator,
    ) -> None:
        super().__init__(args, api_plugin, client, queue)
        self._request_generator = request_generator

    async def run(self) -> None:
        warmup_requests, benchmark_requests = await self._partition_requests(self._request_generator)

        if warmup_requests:
            await self._run_phase(warmup_requests, is_warmup=True)
        await self._run_phase(benchmark_requests, is_warmup=False)

    async def _run_phase(self, requests: List[dict], is_warmup: bool) -> None:
        """Fire all requests in this phase and wait for all to complete."""
        in_flight: set[asyncio.Task] = set()

        for request in requests:
            # Apply Poisson inter-arrival sleep so open-loop semantics are
            # preserved: the interval elapses *before* each dispatch, not
            # during pre-collection.
            if self.args.rate != -1:
                interval = np.random.exponential(1.0 / self.args.rate)
                await asyncio.sleep(interval)

            task = asyncio.create_task(_send_request_open_loop(request, is_warmup, self.queue, self.client))
            in_flight.add(task)

        # Phase barrier: wait for all in-flight requests before returning.
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)
