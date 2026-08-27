import asyncio
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

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
    track_gpu_memory: bool = False,
) -> None:
    async with semaphore:
        benchmark_data = await client.post(request)
    benchmark_data.is_warmup = is_warmup
    if track_gpu_memory:
        benchmark_data.update_gpu_usage()
    await queue.put(benchmark_data)


class ClosedLoopStrategy(BenchmarkStrategy):
    """Closed-loop benchmark strategy.

    Limits the number of in-flight requests to ``args.parallel`` using a
    semaphore.  New requests are only dispatched once a slot becomes available,
    providing back-pressure that prevents the server from being overwhelmed.

    Warmup hand-off
    ---------------
    Warmup and measured requests share a single dispatch loop and a single
    semaphore, and there is deliberately **no barrier** between them.  Draining
    the in-flight requests first would reset server occupancy to zero, so the
    leading ``parallel`` measured requests would again be released at the same
    instant and queue behind one another's prefill - precisely the burst that
    warmup is meant to absorb.  With a continuous hand-off the warmup requests
    absorb it instead: they complete one at a time, and each completion releases
    exactly one measured request into a server that already holds
    ``parallel - 1`` requests mid-flight.  The measured portion therefore starts
    from a steady-state arrival pattern rather than from a synchronised burst,
    which is what makes its TTFT distribution (and hence its percentiles)
    representative of the configured concurrency.

    This requires ``warmup_count >= parallel``, which
    :attr:`~evalscope.perf.arguments.Arguments.warmup_count` enforces for
    closed-loop single-turn runs.  When warmup is disabled the initial burst
    necessarily lands inside the measured window; ``run_benchmark`` warns about
    that rather than silently enabling warmup.
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
        requests = await self._collect_requests(self._request_generator)
        await self._run_phase(requests, duration=self.args.duration)

    async def _run_phase(self, requests: List[Tuple[dict, bool]], duration: Optional[float] = None) -> None:
        """Dispatch ``(request, is_warmup)`` items in order and await completion.

        When ``args.rate`` is configured, request pacing uses absolute-time
        scheduling (see :class:`~evalscope.perf.core.strategies.OpenLoopStrategy`
        for the rationale).  Pre-compute a cumulative delay vector anchored to
        a phase ``start`` timestamp so that event-loop jitter can be absorbed
        instead of accumulating into a slow drift of the realised QPS.

        ``duration``: optional length of the timed window in seconds.  The
        deadline is anchored on the **first measured dispatch**, so warmup never
        consumes the budget.  Once it elapses the dispatch loop stops but
        already in-flight requests are awaited to completion (soft-exit,
        matches trie's semantics).
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

        deadline: Optional[float] = None
        measurement_started = False
        dispatched = 0
        all_tasks: set[asyncio.Task] = set()
        try:
            for i, (request, is_warmup) in enumerate(requests):
                if not is_warmup:
                    # Anchor the timed window on the first measured dispatch;
                    # warmup requests are exempt from the duration cap.
                    if not measurement_started:
                        measurement_started = True
                        deadline = self._compute_deadline(duration)
                    # Duration cap: stop dispatching new requests once the deadline is hit.
                    elif deadline is not None and time.perf_counter() >= deadline:
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
                    await asyncio.gather(*done)

                task = asyncio.create_task(
                    _send_request(semaphore, request, is_warmup, self.queue, self.client, self.track_gpu_memory)
                )
                in_flight.add(task)
                all_tasks.add(task)
                dispatched += 1

            # Phase barrier: wait for all in-flight requests to finish.  Even when
            # the duration deadline has elapsed we let in-flight requests complete
            # (soft exit), matching trie: cap is "stop starting new requests at the
            # deadline", not "kill in-flight work".
            if in_flight:
                if deadline is not None and time.perf_counter() >= deadline:
                    logger.info(f'Duration deadline reached; awaiting {len(in_flight)} in-flight request(s).')
                await asyncio.gather(*in_flight)
        finally:
            for task in all_tasks:
                if not task.done():
                    task.cancel()
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)
