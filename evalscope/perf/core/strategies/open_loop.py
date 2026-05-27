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
            # Warmup ignores --duration (must finish in full before timed window).
            await self._run_phase(warmup_requests, is_warmup=True, deadline=None)
        await self._run_phase(
            benchmark_requests,
            is_warmup=False,
            deadline=self._compute_deadline(self.args.duration),
        )

    async def _run_phase(self, requests: List[dict], is_warmup: bool, deadline: Optional[float] = None) -> None:
        """Fire all requests in this phase and wait for all to complete.

        Uses absolute-time scheduling (à la vLLM ``benchmarks/serve.py``):
        all per-request inter-arrival intervals are pre-computed once, then
        accumulated into absolute wake-up timestamps relative to a phase
        anchor ``start``.  Each iteration sleeps until ``start + delay_ts[i]``
        instead of sleeping a freshly-sampled relative interval.

        Why this matters
        ----------------
        The previous implementation did
        ``await asyncio.sleep(np.random.exponential(1/rate))`` in every
        iteration.  That is *relative* pacing and accumulates drift: any
        event-loop jitter (long-running coroutines, GC, GIL contention from
        SSE chunk handling, etc.) extends each sleep by ``Δ``, and the
        ``Δ`` is never recovered, so the effective send rate decays
        monotonically over time.  Empirically this manifested as server-side
        QPM falling from the target value to ~70-80% over the duration of a
        run, even though the dispatcher *thought* it was still on schedule.

        Absolute-time scheduling self-corrects: if iteration ``i`` is late by
        ``Δ`` ms, iteration ``i+1`` simply computes a smaller (or negative)
        ``sleep_s`` and dispatches immediately, catching up.  The long-run
        average rate stays locked to ``args.rate``.
        """
        in_flight: set[asyncio.Task] = set()
        n = len(requests)
        rate = self.args.rate

        if rate == -1 or n == 0:
            # Unlimited rate: fire all requests as fast as the loop allows.
            for request in requests:
                if deadline is not None and time.perf_counter() >= deadline:
                    logger.info('Duration deadline reached; stopping further dispatches.')
                    break
                task = asyncio.create_task(_send_request_open_loop(request, is_warmup, self.queue, self.client))
                in_flight.add(task)
        else:
            # 1) Sample n Poisson inter-arrival intervals (mean = 1/rate).
            intervals = np.random.exponential(1.0 / rate, size=n)
            # 2) Accumulate into absolute offsets from the phase start.
            delay_ts = np.cumsum(intervals)
            # 3) Re-scale so total phase duration is exactly n / rate.
            #    This eliminates the 1-2% bias that ``np.random.exponential``
            #    accumulates over n samples and keeps the realised QPS
            #    locked to the configured value across runs / seeds.
            target_total_s = n / rate
            if delay_ts[-1] > 0:
                delay_ts *= (target_total_s / delay_ts[-1])

            # 4) Anchor the schedule to absolute monotonic timestamps and
            #    drive dispatch.  We pre-compute ``target_times`` (a numpy
            #    vector) right before the dispatch loop so the per-iteration
            #    cost is a single subtraction + index, not an add + float()
            #    cast.  Keep ``perf_counter()`` adjacent to the loop entry –
            #    do not insert any other awaits between this line and the
            #    loop, otherwise the anchor will skew.
            target_times = delay_ts + time.perf_counter()
            for i, request in enumerate(requests):
                if deadline is not None and time.perf_counter() >= deadline:
                    logger.info(
                        f'Duration deadline reached after dispatching {i}/{n} requests; '
                        'stopping further dispatches.'
                    )
                    break
                sleep_s = target_times[i] - time.perf_counter()
                # Cap the sleep at the remaining time-to-deadline so we don't
                # sleep past the cancellation point.
                if deadline is not None:
                    sleep_s = min(sleep_s, deadline - time.perf_counter())
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                # If sleep_s <= 0 we are behind schedule; dispatch immediately
                # to absorb the drift.
                task = asyncio.create_task(_send_request_open_loop(request, is_warmup, self.queue, self.client))
                in_flight.add(task)

        # Phase barrier: let already-fired requests finish even past the
        # deadline (soft exit, matches trie).  The dispatch loop has already
        # stopped firing new requests once deadline was hit above.
        if in_flight:
            if deadline is not None and time.perf_counter() >= deadline:
                logger.info(f'Duration deadline reached; awaiting {len(in_flight)} in-flight request(s).')
            await asyncio.gather(*in_flight, return_exceptions=True)
