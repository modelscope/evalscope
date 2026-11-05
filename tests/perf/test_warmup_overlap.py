"""Exercise overlap exclusion through the real request stream and result pipeline."""

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from evalscope.perf.benchmark import get_requests
from evalscope.perf.core.pipeline import run_benchmark_pipeline
from evalscope.perf.core.strategies.closed_loop import ClosedLoopStrategy
from evalscope.perf.utils.benchmark_util import BenchmarkData
from tests.perf.test_closed_loop_warmup_handoff import _make_args


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
