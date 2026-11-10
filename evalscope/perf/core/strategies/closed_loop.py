import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Callable, List, Literal, Optional, Tuple

import numpy as np

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies.base import BenchmarkStrategy
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.perf.core.http_client import AioHttpClient
    from evalscope.perf.plugin.api.base import ApiPluginBase

logger = get_logger()


@dataclass
class _MeasurementWindow:
    """Own send-time classification, quota and deadline for one closed-loop run."""

    number: int
    remaining_warmup: int
    duration: Optional[float]
    compute_deadline: Callable[[Optional[float]], Optional[float]]
    exclude_overlap: bool
    measured_sent: int = 0  # Includes requests that later fail.
    queued_candidates: int = 0  # Non-warmup tasks not yet classified at send time.
    deadline: Optional[float] = None
    timer: Optional[asyncio.TimerHandle] = None
    # Stops dispatch and replenishment; already-sent requests still finish.
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    # Wakes the producer when a queued candidate is classified or the window stops.
    send_progress: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def expired(self) -> bool:
        return self.deadline is not None and time.perf_counter() >= self.deadline

    @property
    def closed(self) -> bool:
        return self.stopped.is_set() or self.measured_sent >= self.number or self.expired

    def queue_request(self, is_warmup: bool) -> None:
        if self.exclude_overlap and not is_warmup:
            self.queued_candidates += 1

    def start_request(self, is_warmup: bool) -> Optional[Literal['warmup', 'transition', 'measured']]:
        if is_warmup:
            return 'warmup'
        if self.exclude_overlap:
            self.queued_candidates -= 1
            self.send_progress.set()
            if self.remaining_warmup:
                return 'transition'
            if self.closed:
                return None
        elif self.expired:
            # Queued behind the semaphore until after the deadline: never sent.
            return None

        self.measured_sent += 1
        if self.deadline is None and self.duration is not None:
            self.deadline = self.compute_deadline(self.duration)
            if self.exclude_overlap:
                self.timer = asyncio.get_running_loop().call_later(
                    max(0, self.deadline - time.perf_counter()), self._stop
                )
        if self.exclude_overlap and self.closed:
            self._stop()
        return 'measured'

    def finish_request(self, is_warmup: bool) -> None:
        if is_warmup:
            self.remaining_warmup -= 1

    def _stop(self) -> None:
        self.stopped.set()
        self.send_progress.set()

    def cancel_timer(self) -> None:
        if self.timer is not None:
            self.timer.cancel()


async def _send_request(
    semaphore: asyncio.Semaphore,
    request: dict,
    is_warmup: bool,
    queue: asyncio.Queue,
    client: 'AioHttpClient',
    window: _MeasurementWindow,
    track_gpu_memory: bool = False,
) -> None:
    async with semaphore:
        # Fired when the request actually goes out, not when its task was created.
        phase = window.start_request(is_warmup)
        if phase is None:
            return
        benchmark_data = await client.post(request)
        window.finish_request(is_warmup)
    benchmark_data.is_warmup = is_warmup
    if phase == 'transition':
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

    async def _wait_for_capacity(self, in_flight: set[asyncio.Task]) -> bool:
        if len(in_flight) < self.args.parallel * self.args.in_flight_task_multiplier:
            return False
        done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
        in_flight.difference_update(done)
        await asyncio.gather(*done)
        return True

    async def _dispatch_items(
        self,
        requests: List[Tuple[dict, bool]],
        replacements: Optional[AsyncIterator[Tuple[dict, bool]]],
        window: _MeasurementWindow,
        in_flight: set[asyncio.Task],
    ) -> AsyncIterator[Tuple[dict, bool]]:
        for item in requests:
            yield item
        if replacements is None:
            return
        while not window.closed:
            # Already queued candidates can still fill the measured quota.
            if window.queued_candidates >= window.number - window.measured_sent:
                window.send_progress.clear()
                await window.send_progress.wait()
                continue
            if await self._wait_for_capacity(in_flight):
                continue
            replacement = asyncio.ensure_future(anext(replacements))
            closed = asyncio.create_task(window.stopped.wait())
            try:
                await asyncio.wait((replacement, closed), return_when=asyncio.FIRST_COMPLETED)
                if window.closed:
                    return
                item = replacement.result()
            except StopAsyncIteration:
                return
            finally:
                replacement.cancel()
                closed.cancel()
                await asyncio.gather(replacement, closed, return_exceptions=True)
            yield item

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
        in_flight: set[asyncio.Task] = set()
        n = len(requests)
        rate = self.args.rate
        window = _MeasurementWindow(
            number=self.args.number,
            remaining_warmup=sum(is_warmup for _, is_warmup in requests),
            duration=duration,
            compute_deadline=self._compute_deadline,
            exclude_overlap=replacements is not None,
        )

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

        dispatched = 0
        last_target = None
        try:
            async for request, is_warmup in self._dispatch_items(requests, replacements, window, in_flight):
                if window.exclude_overlap and not is_warmup and window.closed:
                    break
                # Duration cap: stop dispatching once the deadline is hit.  It is
                # ``None`` until a measured request has been sent, so warmup is exempt.
                if window.expired:
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
                    if window.deadline is not None:
                        sleep_s = min(sleep_s, window.deadline - time.perf_counter())
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)

                # Keep the number of scheduled tasks bounded to avoid OOM.
                await self._wait_for_capacity(in_flight)

                if window.expired:
                    logger.info(
                        f'Duration deadline reached after dispatching {dispatched}/{n} requests; '
                        'stopping further dispatches.'
                    )
                    break

                window.queue_request(is_warmup)
                task = asyncio.create_task(
                    _send_request(
                        semaphore,
                        request,
                        is_warmup,
                        self.queue,
                        self.client,
                        window,
                        self.track_gpu_memory,
                    )
                )
                in_flight.add(task)
                dispatched += 1

            # Phase barrier: wait for all in-flight requests to finish.  Even when
            # the duration deadline has elapsed we let in-flight requests complete
            # (soft exit), matching trie: cap is "stop starting new requests at the
            # deadline", not "kill in-flight work".
            if in_flight:
                if window.expired:
                    logger.info(f'Duration deadline reached; awaiting {len(in_flight)} in-flight request(s).')
                await asyncio.gather(*in_flight)
            if window.exclude_overlap and not window.closed:
                raise RuntimeError(
                    f'Request source ended after {window.measured_sent}/{window.number} measured requests.'
                )
        finally:
            window.cancel_timer()
            for task in in_flight:
                if not task.done():
                    task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
