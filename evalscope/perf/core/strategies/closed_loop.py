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


async def _send_request(
    semaphore: asyncio.Semaphore,
    request: dict,
    is_warmup: bool,
    queue: asyncio.Queue,
    client: 'AioHttpClient',
) -> None:
    async with semaphore:
        benchmark_data = await client.post(request)
    benchmark_data.is_warmup = is_warmup
    benchmark_data.update_gpu_usage()
    await queue.put(benchmark_data)


class ClosedLoopStrategy(BenchmarkStrategy):
    """Closed-loop benchmark strategy.

    Limits the number of in-flight requests to ``args.parallel`` using a
    semaphore.  New requests are only dispatched once a slot becomes available,
    providing back-pressure that prevents the server from being overwhelmed.
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
        """Dispatch one phase of requests and wait for all to complete."""
        semaphore = asyncio.Semaphore(self.args.parallel)
        max_in_flight = self.args.parallel * self.args.in_flight_task_multiplier
        in_flight: set[asyncio.Task] = set()

        for request in requests:
            # Apply Poisson inter-arrival sleep when rate is set.
            # Closed-loop combines back-pressure (semaphore) with arrival-rate
            # control (rate), allowing both concurrency cap and request pacing.
            if self.args.rate != -1:
                interval = np.random.exponential(1.0 / self.args.rate)
                await asyncio.sleep(interval)

            # Keep the number of scheduled tasks bounded to avoid OOM.
            if len(in_flight) >= max_in_flight:
                done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                in_flight = pending

            task = asyncio.create_task(_send_request(semaphore, request, is_warmup, self.queue, self.client))
            in_flight.add(task)

        # Phase barrier: wait for all in-flight requests before returning.
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)
