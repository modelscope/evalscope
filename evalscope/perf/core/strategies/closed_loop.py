import asyncio
import time
from typing import TYPE_CHECKING, AsyncIterator, Callable, List, Optional, Tuple

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
    on_send: Optional[Callable[[], Optional[bool]]] = None,
    on_warmup_complete: Optional[Callable[[], None]] = None,
) -> None:
    async with semaphore:
        # Fired when the request actually goes out, not when its task was created.
        is_transition = on_send() if on_send is not None else False
        if is_transition is None:
            return
        benchmark_data = await client.post(request)
        if is_warmup and on_warmup_complete is not None:
            on_warmup_complete()
    benchmark_data.is_warmup = is_warmup
    if is_transition:
        benchmark_data.is_transition = True
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
    Warmup and measured requests share one dispatch loop and one semaphore, with
    deliberately **no barrier** between them.  Re-introducing one would drain
    server occupancy to zero, so the leading ``parallel`` measured requests would
    again be released together and queue behind one another's prefill - the very
    burst warmup exists to absorb.  Instead each warmup completion releases
    exactly one measured request into a server still holding ``parallel - 1``
    requests.

    Absorbing the whole burst needs ``warmup_count >= parallel``; smaller values
    cover fewer slots and ``run_benchmark`` warns about it.
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
        if not self.args.exclude_warmup_overlap or self.args.warmup_count == 0:
            requests = await self._collect_requests(self._request_generator)
            await self._run_phase(requests, duration=self.args.duration)
            return

        requests = []
        try:
            async for item in self._request_generator:
                requests.append(item)
                if len(requests) >= self.args.total_count:
                    break
            await self._run_phase(requests, duration=self.args.duration, replacements=self._request_generator)
        finally:
            await self._request_generator.aclose()

    async def _run_phase(
        self,
        requests: List[Tuple[dict, bool]],
        duration: Optional[float] = None,
        replacements: Optional[AsyncIterator[Tuple[dict, bool]]] = None,
    ) -> None:
        """Dispatch ``(request, is_warmup)`` items in order and await completion.

        When ``args.rate`` is configured, request pacing uses absolute-time
        scheduling (see :class:`~evalscope.perf.core.strategies.OpenLoopStrategy`
        for the rationale).  Pre-compute a cumulative delay vector anchored to
        a phase ``start`` timestamp so that event-loop jitter can be absorbed
        instead of accumulating into a slow drift of the realised QPS.

        ``duration``: optional length of the timed window in seconds.  The
        deadline is anchored on the moment the **first measured request is sent**,
        so warmup does not consume the budget.  Once it elapses the dispatch loop
        stops but already in-flight requests are awaited to completion (soft-exit,
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
                delay_ts *= target_total_s / delay_ts[-1]
            # Keep ``perf_counter()`` adjacent to the loop entry – do not
            # insert any other awaits between this line and the loop,
            # otherwise the anchor will skew.
            target_times = delay_ts + time.perf_counter()

        deadline: Optional[float] = None
        deadline_timer: Optional[asyncio.TimerHandle] = None
        measurement_closed = asyncio.Event()
        send_progress = asyncio.Event()

        def _close_measurement() -> None:
            measurement_closed.set()
            send_progress.set()

        def _arm_deadline() -> None:
            """Start the timed window on the first measured request that goes out.

            The dispatch loop is bounded by ``max_in_flight``, not by the
            semaphore, so it queues measured tasks while warmup still holds every
            slot; arming on dispatch would start the clock at ~t0.
            """
            nonlocal deadline, deadline_timer
            if deadline is None and duration is not None:
                deadline = self._compute_deadline(duration)
                if exclude_overlap:
                    deadline_timer = asyncio.get_running_loop().call_later(
                        max(0, deadline - time.perf_counter()), _close_measurement
                    )

        exclude_overlap = replacements is not None
        remaining_warmup = sum(is_warmup for _, is_warmup in requests) if exclude_overlap else 0
        measured = 0
        pending_candidates = 0

        def _measurement_closed() -> bool:
            return (
                measurement_closed.is_set()
                or measured >= self.args.number
                or (deadline is not None and time.perf_counter() >= deadline)
            )

        def _classify_send() -> Optional[bool]:
            nonlocal measured, pending_candidates
            if exclude_overlap:
                pending_candidates -= 1
                send_progress.set()
            if remaining_warmup:
                return True
            if exclude_overlap and _measurement_closed():
                return None
            measured += 1
            _arm_deadline()
            if exclude_overlap and _measurement_closed():
                _close_measurement()
            return False

        def _finish_warmup() -> None:
            nonlocal remaining_warmup
            remaining_warmup -= 1

        async def _dispatch_items() -> AsyncIterator[Tuple[dict, bool]]:
            nonlocal in_flight
            for item in requests:
                yield item
            if replacements is None:
                return
            while not _measurement_closed():
                # Already queued candidates can still fill the measured quota.
                if pending_candidates >= self.args.number - measured:
                    send_progress.clear()
                    await send_progress.wait()
                    continue
                if len(in_flight) >= max_in_flight:
                    done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                    await asyncio.gather(*done)
                    continue
                replacement = asyncio.ensure_future(anext(replacements))
                closed = asyncio.create_task(measurement_closed.wait())
                try:
                    await asyncio.wait((replacement, closed), return_when=asyncio.FIRST_COMPLETED)
                    if _measurement_closed():
                        return
                    item = replacement.result()
                except StopAsyncIteration:
                    return
                finally:
                    replacement.cancel()
                    closed.cancel()
                    await asyncio.gather(replacement, closed, return_exceptions=True)
                yield item

        dispatched = 0
        last_target = None
        try:
            async for request, is_warmup in _dispatch_items():
                if exclude_overlap and not is_warmup and _measurement_closed():
                    break
                # Duration cap: stop dispatching once the deadline is hit.  It is
                # ``None`` until a measured request has been sent, so warmup is exempt.
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
                    if dispatched < n:
                        last_target = target_times[dispatched]
                    else:
                        last_target += float(np.random.exponential(1.0 / rate))
                    sleep_s = last_target - time.perf_counter()
                    if deadline is not None:
                        sleep_s = min(sleep_s, deadline - time.perf_counter())
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)

                # Keep the number of scheduled tasks bounded to avoid OOM.
                if len(in_flight) >= max_in_flight:
                    done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                    in_flight = pending
                    await asyncio.gather(*done)

                if deadline is not None and time.perf_counter() >= deadline:
                    logger.info(
                        f'Duration deadline reached after dispatching {dispatched}/{n} requests; '
                        'stopping further dispatches.'
                    )
                    break

                if exclude_overlap and not is_warmup:
                    pending_candidates += 1
                task = asyncio.create_task(
                    _send_request(
                        semaphore,
                        request,
                        is_warmup,
                        self.queue,
                        self.client,
                        self.track_gpu_memory,
                        on_send=None if is_warmup else _classify_send,
                        on_warmup_complete=_finish_warmup if exclude_overlap else None,
                    )
                )
                in_flight.add(task)
                dispatched += 1

            # Phase barrier: wait for all in-flight requests to finish.  Even when
            # the duration deadline has elapsed we let in-flight requests complete
            # (soft exit), matching trie: cap is "stop starting new requests at the
            # deadline", not "kill in-flight work".
            if in_flight:
                if deadline is not None and time.perf_counter() >= deadline:
                    logger.info(f'Duration deadline reached; awaiting {len(in_flight)} in-flight request(s).')
                await asyncio.gather(*in_flight)
            if exclude_overlap and not _measurement_closed():
                raise RuntimeError(f'Request source ended after {measured}/{self.args.number} measured requests.')
        finally:
            if deadline_timer is not None:
                deadline_timer.cancel()
            for task in in_flight:
                if not task.done():
                    task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
