"""Closed-loop warmup hand-off.

The measured portion of a closed-loop run must start from a saturated server:
if the dispatcher drained its in-flight requests after warmup, the leading
``parallel`` measured requests would be released simultaneously against an idle
server and their TTFT would carry a start-up burst that never recurs, polluting
the reported percentiles.  These tests pin the hand-off behaviour rather than
timings, so they need no real endpoint.
"""
import asyncio
import os
import sqlite3
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import pytest

from evalscope.perf import benchmark as benchmark_module
from evalscope.perf.arguments import Arguments
from evalscope.perf.benchmark import run_benchmark
from evalscope.perf.core.strategies.closed_loop import ClosedLoopStrategy
from tests.perf.test_async_lifecycle import LocalOpenAIServer


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
    """Client that records server occupancy at the moment each request starts.

    Each request completes after a distinct number of event-loop ticks so the
    completion order is deterministic and the warmup requests finish one at a
    time, mirroring the output-length spread of a real server.
    """

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
            # Strictly increasing service times so completions never coincide,
            # mirroring the output-length spread of a real server: slots free up
            # one at a time and each frees exactly one waiting request.
            for _ in range(2 * (seq + 1)):
                await asyncio.sleep(0)
        finally:
            self.in_flight -= 1
        return SimpleNamespace(is_warmup=False)


def _run(args: Arguments, requests: List[Tuple[dict, bool]]) -> _RecordingClient:
    """Drive ClosedLoopStrategy over ``requests`` and return the client."""
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


# --- hand-off behaviour ---------------------------------------------------


def test_measured_portion_starts_without_draining() -> None:
    """The first measured request must find the other slots still busy."""
    parallel = 4
    args = _make_args(number=6, parallel=parallel)
    client = _run(args, _stream(warmup=parallel, measured=6))

    warmup_ids = {i for i in range(parallel)}
    measured = [e for e in client.events if e['request']['id'] not in warmup_ids]
    assert measured, 'no measured request was dispatched'
    first_measured = min(measured, key=lambda e: e['order'])
    assert first_measured['occupancy_at_start'] == parallel - 1
    # Occupancy never returns to zero either: only the very first request of the
    # run starts against an idle server.
    later = [e['occupancy_at_start'] for e in client.events if e['order'] > 0]
    assert later and all(occupancy > 0 for occupancy in later)


def test_dispatch_order_is_warmup_then_measured() -> None:
    args = _make_args(number=4, parallel=2)
    stream = _stream(warmup=2, measured=4)
    client = _run(args, stream)

    ids = [e['request']['id'] for e in sorted(client.events, key=lambda e: e['order'])]
    assert ids == [0, 1, 2, 3, 4, 5]


def test_all_requests_dispatched_without_warmup() -> None:
    """warmup_num=0 keeps the previous behaviour: every request is measured."""
    args = _make_args(number=5, parallel=2)
    client = _run(args, _stream(warmup=0, measured=5))
    assert len(client.events) == 5


def test_partial_warmup_degrades_gracefully() -> None:
    """Fewer warmup requests than slots still runs; it just covers less.

    The uncovered slots are filled by measured requests released alongside the
    warmup ones, so the burst is only partly absorbed - which is exactly what
    the warning tells the user.
    """
    parallel = 4
    args = _make_args(number=6, parallel=parallel)
    client = _run(args, _stream(warmup=2, measured=6))

    assert len(client.events) == 8
    # The opening cohort still fills every slot before anything completes, so
    # it contains 2 warmup plus 2 measured requests.
    opening = sorted(client.events, key=lambda e: e['order'])[:parallel]
    assert [e['occupancy_at_start'] for e in opening] == list(range(parallel))
    assert sum(1 for e in opening if e['request']['id'] >= 2) == 2


# --- duration anchoring ---------------------------------------------------


def test_duration_exempts_warmup_and_caps_measured() -> None:
    """Warmup is exempt from --duration; the budget starts at measurement."""
    parallel, measured = 2, 20
    args = _make_args(number=measured, parallel=parallel, duration=0.0)
    client = _run(args, _stream(warmup=parallel, measured=measured))

    dispatched_ids = {e['request']['id'] for e in client.events}
    # All warmup requests ran even though the budget is exhausted instantly.
    assert {0, 1}.issubset(dispatched_ids)
    # The window is armed by the first measured send, so at least one measured
    # request always goes out, and the cap then stops the rest.
    measured_ids = {i for i in dispatched_ids if i >= parallel}
    assert 1 <= len(measured_ids) < measured


def test_duration_budget_excludes_warmup_service_time() -> None:
    """The window is armed on the first measured *send*, not on task creation.

    Task creation is bounded by ``max_in_flight``, not by the semaphore, so
    measured tasks exist long before a slot frees up.  Arming on creation would
    start the clock at ~t0 and silently bill the warmup service time to
    ``--duration``.
    """
    parallel, service_s = 2, 0.05

    class _TimedClient:

        def __init__(self) -> None:
            self.first_warmup_done_at: Optional[float] = None

        async def post(self, request: Dict[str, Any]) -> SimpleNamespace:
            await asyncio.sleep(service_s)
            if request['id'] < parallel and self.first_warmup_done_at is None:
                self.first_warmup_done_at = time.perf_counter()
            return SimpleNamespace(is_warmup=False)

    class _AnchorSpy(ClosedLoopStrategy):
        """Records the moment the timed window is armed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.anchored_at: Optional[float] = None

        def _compute_deadline(self, duration: Optional[float]) -> Optional[float]:
            self.anchored_at = time.perf_counter()
            return super()._compute_deadline(duration)

    args = _make_args(number=6, parallel=parallel, duration=10.0, warmup_num=parallel)
    client = _TimedClient()
    requests = _stream(warmup=parallel, measured=6)

    async def generator() -> AsyncIterator[Tuple[dict, bool]]:
        for item in requests:
            yield item

    async def main() -> _AnchorSpy:
        strategy = _AnchorSpy(args, None, client, asyncio.Queue(), generator())
        await strategy.run()
        return strategy

    strategy = asyncio.run(main())

    assert strategy.anchored_at is not None
    assert client.first_warmup_done_at is not None
    # Arming happens no earlier than the first freed slot, so none of the warmup
    # service time is charged to the measured window.
    assert strategy.anchored_at >= client.first_warmup_done_at


def test_duration_none_dispatches_everything() -> None:
    args = _make_args(number=4, parallel=2)
    client = _run(args, _stream(warmup=2, measured=4))
    assert len(client.events) == 6


# --- user-facing diagnostics ---------------------------------------------


class _RecordingLogger:

    def __init__(self) -> None:
        self.infos: List[str] = []
        self.warnings: List[str] = []

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _capture_handoff_log(monkeypatch: pytest.MonkeyPatch, args: Arguments) -> _RecordingLogger:
    recorder = _RecordingLogger()
    monkeypatch.setattr(benchmark_module, 'logger', recorder)
    benchmark_module._log_warmup_handoff(args)
    return recorder


def test_warns_when_warmup_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _capture_handoff_log(monkeypatch, _make_args(number=100, parallel=8))
    assert len(recorder.warnings) == 1
    assert 'Warmup is disabled' in recorder.warnings[0]
    assert '--warmup-num 8' in recorder.warnings[0]
    assert '16' in recorder.warnings[0]


def test_warns_when_warmup_smaller_than_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _capture_handoff_log(monkeypatch, _make_args(number=100, parallel=8, warmup_num=3))
    assert len(recorder.warnings) == 1
    assert 'only 3 of the 8 concurrency slots' in recorder.warnings[0]
    assert '5 measured requests' in recorder.warnings[0]


def test_no_warning_for_single_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _capture_handoff_log(monkeypatch, _make_args(number=100, parallel=1))
    assert recorder.warnings == []


def test_silent_when_warmup_covers_every_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _capture_handoff_log(monkeypatch, _make_args(number=100, parallel=8, warmup_num=8))
    assert recorder.warnings == []
    assert recorder.infos == []


# --- end to end -----------------------------------------------------------


def test_end_to_end_warmup_hits_server_but_stays_out_of_results(tmp_path) -> None:
    """Warmup requests reach the server yet never land in the result DB."""
    parallel, number = 3, 5
    output_dir = tmp_path / 'e2e'
    output_dir.mkdir()

    def make_args(port: int) -> Arguments:
        args = Arguments(
            model='test-model',
            api='openai',
            number=number,
            parallel=parallel,
            rate=-1,
            url=f'http://127.0.0.1:{port}/v1/chat/completions',
            prompt='hello',
            stream=False,
            no_test_connection=True,
            outputs_dir=str(output_dir),
            warmup_num=parallel,
        )
        args.number = number
        args.parallel = parallel
        args.rate = -1
        args.outputs_dir = str(output_dir)
        return args

    async def run() -> int:
        server = LocalOpenAIServer()
        port = await server.start()
        try:
            args = make_args(port)
            assert args.warmup_count == parallel
            assert args.total_count == parallel + number
            await run_benchmark(args)
            return server.request_count
        finally:
            await server.close()

    request_count = asyncio.run(run())
    assert request_count == parallel + number

    with sqlite3.connect(os.path.join(str(output_dir), 'benchmark_data.db')) as con:
        rows = con.execute('SELECT COUNT(*) FROM result').fetchone()[0]
    assert rows == number
