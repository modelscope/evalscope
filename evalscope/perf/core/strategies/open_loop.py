import asyncio
from typing import TYPE_CHECKING

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies.base import BenchmarkStrategy
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.perf.core.http_client import AioHttpClient
    from evalscope.perf.plugin.api.base import ApiPluginBase

logger = get_logger()


async def _send_request_open_loop(
    request: dict,
    queue: asyncio.Queue,
    client: 'AioHttpClient',
) -> None:
    """Open-loop send: fires immediately regardless of in-flight count."""
    benchmark_data = await client.post(request)
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
        in_flight: set[asyncio.Task] = set()

        async for request in self._request_generator:
            task = asyncio.create_task(_send_request_open_loop(request, self.queue, self.client))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)
