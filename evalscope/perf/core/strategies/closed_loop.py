import asyncio
import numpy as np
import time
from typing import TYPE_CHECKING, List, Optional

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
            # Warmup ignores --duration; it must finish in full before the
            # timed benchmark window begins.
            await self._run_phase(warmup_requests, is_warmup=True, deadline=None)
        await self._run_phase(
            benchmark_requests,
            is_warmup=False,
            deadline=self._compute_deadline(self.args.duration),
        )

    async def _run_phase(self, requests: List[dict], is_warmup: bool, deadline: Optional[float] = None) -> None:
        """Dispatch one phase of requests and wait for all to complete.

        When ``args.rate`` is configured, request pacing uses absolute-time
        scheduling (see :class:`~evalscope.perf.core.strategies.OpenLoopStrategy`
        for the rationale).  Pre-compute a cumulative delay vector anchored to
        a phase ``start`` timestamp so that event-loop jitter can be absorbed
        instead of accumulating into a slow drift of the realised QPS.

        ``deadline``: optional ``time.perf_counter()`` timestamp; when set, the
        dispatch loop exits on reaching it but already in-flight requests are
        awaited to completion (soft-exit, matches trie's semantics).
        """
        semaphore = asyncio.Semaphore(self.args.parallel)
        max_in_flight = self.args.parallel * self.args.in_flight_task_multiplier
        in_flight: set[asyncio.Task] = set()
        n = len(requests)
        rate = self.args.rate

        # Pre-compute absolute dispatch timestamps when pacing is enabled.
        # ``target_times`` is anchored just before the dispatch loop so that
        # each iteration only needs a single subtraction + index lookup.
        target_times = None
        if rate != -1 and n > 0:
            intervals = np.random.exponential(1.0 / rate, size=n)
            delay_ts = np.cumsum(intervals)
            target_total_s = n / rate
            if delay_ts[-1] > 0:
                delay_ts *= (target_total_s / delay_ts[-1])
            # Keep ``perf_counter()`` adjacent to the loop entry – do not
            # insert any other awaits between this line and the loop,
            # otherwise the anchor will skew.
            target_times = delay_ts + time.perf_counter()

        dispatched = 0
        for i, request in enumerate(requests):
            # Duration cap: stop dispatching new requests once the deadline is hit.
            if deadline is not None and time.perf_counter() >= deadline:
                logger.info(
                    f'Duration deadline reached after dispatching {dispatched}/{n} requests; '
                    'stopping further dispatches.'
                )
                break

            # Sleep until the absolute target dispatch time (drift-corrected).
            # Cap the sleep at the remaining time-to-deadline so we don't sleep
            # past the cancellation point.
            if target_times is not None:
                sleep_s = target_times[i] - time.perf_counter()
                if deadline is not None:
                    sleep_s = min(sleep_s, deadline - time.perf_counter())
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

            # Keep the number of scheduled tasks bounded to avoid OOM.
            if len(in_flight) >= max_in_flight:
                done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                in_flight = pending

            task = asyncio.create_task(_send_request(semaphore, request, is_warmup, self.queue, self.client))
            in_flight.add(task)
            dispatched += 1

        # Phase barrier: wait for all in-flight requests to finish.  Even when
        # the duration deadline has elapsed we let in-flight requests complete
        # (soft exit), matching trie: cap is "stop starting new requests at the
        # deadline", not "kill in-flight work".
        if in_flight:
            if deadline is not None and time.perf_counter() >= deadline:
                logger.info(f'Duration deadline reached; awaiting {len(in_flight)} in-flight request(s).')
            await asyncio.gather(*in_flight, return_exceptions=True)
