"""Exercise overlap exclusion through the real request stream and result pipeline."""

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterator, Optional

import pytest

from evalscope.perf import benchmark as benchmark_module
from evalscope.perf.benchmark import get_requests
from evalscope.perf.core.pipeline import run_benchmark_pipeline
from evalscope.perf.core.strategies import closed_loop as closed_loop_module
from evalscope.perf.core.strategies.closed_loop import ClosedLoopStrategy
from evalscope.perf.utils.benchmark_util import BenchmarkData
from tests.perf.test_closed_loop_warmup_handoff import _make_args, _run, _stream


@pytest.mark.parametrize(('enabled', 'warmup', 'duration', 'expected'), [
    (False, 2, None, 3),
    (True, 0, None, 3),
    (True, 2, None, 3),
    (True, 2, 0.0, 1),
])
def test_overlap_result_partition_and_replenishment(
    tmp_path: Path, enabled: bool, warmup: int, duration: Optional[float], expected: int
) -> None:
    args = _make_args(
        number=3, parallel=2, warmup_num=warmup, prompt='test',
        exclude_warmup_overlap=enabled, outputs_dir=str(tmp_path), duration=duration,
    )
    starts: dict[int, float] = {}
    ends: dict[int, float] = {}

    async def run() -> None:
        release_warmup = asyncio.Event()
        # Hold one slot through five transitions. The initial stream alone is
        # too short to release it when replenishment is enabled.
        release_at = 6 if enabled and warmup else 4
        in_flight = 0

        class Client:
            async def post(self, request: dict[str, Any]) -> BenchmarkData:
                nonlocal in_flight
                index = len(starts)
                starts[index] = time.perf_counter()
                in_flight += 1
                assert in_flight <= 2
                if index == 0 and warmup:
                    await release_warmup.wait()
                else:
                    await asyncio.sleep(0)
                ends[index] = time.perf_counter()
                in_flight -= 1
                if index == release_at:
                    release_warmup.set()
                return BenchmarkData(
                    request=json.dumps(request), request_id=str(index),
                    start_time=starts[index], completed_time=ends[index],
                    query_latency=ends[index]-starts[index], first_chunk_latency=0.001,
                    prompt_tokens=8, completion_tokens=2, success=index != 3,
                )

        api = SimpleNamespace(build_request=lambda prompt: {'prompt': prompt})
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        strategy = ClosedLoopStrategy(args, api, Client(), queue, get_requests(args, api))
        await asyncio.wait_for(run_benchmark_pipeline(strategy.run(), queue, args, api), timeout=5)

    asyncio.run(run())
    with sqlite3.connect(tmp_path / 'benchmark_data.db') as con:
        result = con.execute('SELECT request_id, start_time FROM result ORDER BY start_time').fetchall()
        assert len(result) == expected
        if not enabled:
            assert con.execute("SELECT 1 FROM sqlite_master WHERE name='transition'").fetchone() is None
            assert len(starts) == warmup + expected
            return
        transition = con.execute('SELECT request_id, start_time, success FROM transition').fetchall()
        if not warmup:
            assert not transition
            assert len(starts) == expected
            return
        assert len(transition) >= 5
        assert any(success == 0 for _, _, success in transition)
        assert all(start < ends[0] for _, start, _ in transition)
        assert all(start >= ends[0] for _, start in result)
        assert len(starts) == warmup + len(transition) + expected


@pytest.mark.parametrize('mode', ['open_loop', 'multi_turn'])
def test_overlap_rejects_unsupported_modes(mode: str) -> None:
    with pytest.raises(ValueError, match='requires single-turn closed-loop'):
        _make_args(exclude_warmup_overlap=True, **{mode: True})


def test_zero_measured_requests_still_runs_warmup() -> None:
    args = _make_args(number=0, parallel=2, warmup_num=2, exclude_warmup_overlap=True)
    client = _run(args, _stream(warmup=2, measured=0))
    assert [event['request']['id'] for event in client.events] == [0, 1]


def test_random_replenishment_runs_off_event_loop_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    main_thread = threading.get_ident()
    generation_threads: list[int] = []
    replacement_started = threading.Event()
    allow_replacement = threading.Event()

    class Dataset:

        def __init__(self, args: Any) -> None:
            pass

        def supports_parallel_message_generation(self, total_count: int) -> bool:
            return False

        def build_messages(self) -> Iterator[str]:
            yield from ['initial-a', 'initial-b']

        def build_messages_parallel(
            self, total_count: int, workers: int, *, rng: Any, start_index: int, stop: threading.Event
        ) -> list[str]:
            assert rng is not None
            generation_threads.append(threading.get_ident())
            replacement_started.set()
            assert allow_replacement.wait(timeout=2)
            return ['replacement']

    monkeypatch.setattr(benchmark_module.DatasetRegistry, 'get_class', lambda _: Dataset)
    args = _make_args(
        number=1, parallel=1, warmup_num=1, dataset='random', prompt=None,
        prefix_count=1, exclude_warmup_overlap=True,
    )
    api = SimpleNamespace(build_request=lambda messages: {'prompt': messages})

    async def run() -> None:
        stream = get_requests(args, api)
        try:
            assert (await anext(stream))[0] == {'prompt': 'initial-a'}
            assert (await anext(stream))[0] == {'prompt': 'initial-b'}
            assert not replacement_started.is_set()
            replacement = asyncio.create_task(anext(stream))
            assert await asyncio.to_thread(replacement_started.wait, 1)
            allow_replacement.set()
            assert (await replacement)[0] == {'prompt': 'replacement'}
        finally:
            allow_replacement.set()
            await stream.aclose()

    asyncio.run(run())
    assert generation_threads and generation_threads[0] != main_thread


def test_transition_task_cleanup_stays_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    gather_sizes: list[int] = []
    real_gather = asyncio.gather

    def recording_gather(*aws: Any, **kwargs: Any) -> Any:
        gather_sizes.append(len(aws))
        return real_gather(*aws, **kwargs)

    monkeypatch.setattr(closed_loop_module.asyncio, 'gather', recording_gather)
    args = _make_args(
        number=1, parallel=2, warmup_num=1, exclude_warmup_overlap=True,
        in_flight_task_multiplier=2,
    )

    async def replacements() -> AsyncIterator[tuple[dict, bool]]:
        while True:
            yield {'id': 'replacement'}, False

    async def run() -> None:
        release_warmup = asyncio.Event()
        enough_transitions = asyncio.Event()
        transition_count = 0

        class Client:

            def __init__(self) -> None:
                self.index = 0

            async def post(self, request: dict[str, Any]) -> SimpleNamespace:
                nonlocal transition_count
                index = self.index
                self.index += 1
                if index == 0:
                    await release_warmup.wait()
                else:
                    await asyncio.sleep(0)
                    transition_count += 1
                    if transition_count >= 20:
                        enough_transitions.set()
                return SimpleNamespace(is_warmup=False)

        queue: asyncio.Queue = asyncio.Queue()
        strategy = ClosedLoopStrategy(args, None, Client(), queue, None)
        phase = asyncio.create_task(strategy._run_phase(
            [({'id': 'warmup'}, True), ({'id': 'initial'}, False)],
            replacements=replacements(),
        ))
        await asyncio.wait_for(enough_transitions.wait(), timeout=2)
        release_warmup.set()
        await asyncio.wait_for(phase, timeout=2)

    asyncio.run(run())
    assert max(gather_sizes) <= args.parallel * args.in_flight_task_multiplier


def test_queued_candidate_fills_quota_without_replenishment() -> None:
    args = _make_args(number=1, parallel=1, warmup_num=1, exclude_warmup_overlap=True, duration=1)

    async def requests() -> AsyncIterator[tuple[dict, bool]]:
        yield {'id': 'warmup'}, True
        yield {'id': 'measured'}, False
        pytest.fail('A queued request can fill the quota without replenishment.')

    async def run() -> None:
        class Client:
            async def post(self, request: dict) -> SimpleNamespace:
                return SimpleNamespace(id=request['id'], is_warmup=False)

        queue: asyncio.Queue = asyncio.Queue()
        strategy = ClosedLoopStrategy(args, None, Client(), queue, requests())
        await asyncio.wait_for(strategy.run(), timeout=1)
        assert [queue.get_nowait().id for _ in range(queue.qsize())] == ['warmup', 'measured']

    asyncio.run(run())


def test_duration_stops_blocked_random_replenishment(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_stopped = threading.Event()
    args = _make_args(
        number=2, parallel=2, warmup_num=1, dataset='random', prompt=None,
        prefix_count=1, exclude_warmup_overlap=True, duration=0.05,
    )

    async def run() -> None:
        loop = asyncio.get_running_loop()
        replacement_started = asyncio.Event()

        class Dataset:

            def __init__(self, args: Any) -> None:
                pass

            def supports_parallel_message_generation(self, total_count: int) -> bool:
                return False

            def build_messages(self) -> Iterator[str]:
                yield from ['warmup', 'transition', 'measured']

            def build_messages_parallel(
                self, total_count: int, workers: int, *, rng: Any, start_index: int, stop: threading.Event
            ) -> list[str]:
                loop.call_soon_threadsafe(replacement_started.set)
                if stop.wait(timeout=2):
                    worker_stopped.set()
                raise RuntimeError('Unused replacement failed')

        class Client:

            async def post(self, request: dict[str, Any]) -> SimpleNamespace:
                if request['prompt'] in ('warmup', 'transition'):
                    await replacement_started.wait()
                return SimpleNamespace(is_warmup=False, is_transition=False, prompt=request['prompt'])

        monkeypatch.setattr(benchmark_module.DatasetRegistry, 'get_class', lambda _: Dataset)
        api = SimpleNamespace(build_request=lambda messages: {'prompt': messages})
        queue: asyncio.Queue = asyncio.Queue()
        strategy = ClosedLoopStrategy(args, api, Client(), queue, get_requests(args, api))
        await asyncio.wait_for(strategy.run(), timeout=1)
        assert worker_stopped.is_set()
        rows = [queue.get_nowait() for _ in range(queue.qsize())]
        assert [row.prompt for row in rows if not row.is_warmup and not row.is_transition] == ['measured']

    asyncio.run(run())
