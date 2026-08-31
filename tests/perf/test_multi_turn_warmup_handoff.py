"""Multi-turn warmup hand-off tests.

Warmup conversations must occupy the server until measured conversations take
over.  A phase barrier would drain occupancy to zero and release the opening
measured cohort against an idle server, which is the defect tracked by #1648.
"""
import asyncio
import json
import os
import sqlite3
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.pipeline import run_benchmark_pipeline
from evalscope.perf.core.strategies import multi_turn as multi_turn_module
from evalscope.perf.core.strategies.multi_turn import MultiTurnStrategy
from evalscope.perf.plugin.datasets.base import Conversation, Turn
from evalscope.perf.utils.benchmark_util import BenchmarkData


def _make_args(**kwargs: Any) -> Arguments:
    number = kwargs.pop('number', 6)
    parallel = kwargs.pop('parallel', 3)
    warmup_num = kwargs.pop('warmup_num', 0)
    args = Arguments(
        model='test-model',
        api='openai',
        number=number,
        parallel=parallel,
        rate=-1,
        warmup_num=warmup_num,
        multi_turn=True,
        **kwargs,
    )
    args.number = number
    args.parallel = parallel
    args.rate = -1
    return args


def _conversations(count: int) -> List[Conversation]:
    return [[Turn(messages=[{'role': 'user', 'content': f'conv-{i}'}])] for i in range(count)]


class _ApiPlugin:

    def build_request(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {'messages': messages, 'stream': True}

    def parse_responses(self, response_messages: List[Any], request: Optional[str] = None) -> tuple[int, int]:
        return 1, 1


class _RecordingClient:
    """Fake server that records occupancy when each request starts."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.events: List[Dict[str, Any]] = []
        self._seq = 0

    async def post(self, request: Dict[str, Any]) -> BenchmarkData:
        seq = self._seq
        self._seq += 1
        conv_id = request['messages'][0]['content']
        start_time = time.perf_counter()
        self.events.append({
            'conv_id': conv_id,
            'occupancy_at_start': self.in_flight,
            'order': seq,
        })
        self.in_flight += 1
        try:
            for _ in range(2 * (seq + 1)):
                await asyncio.sleep(0)
        finally:
            self.in_flight -= 1
        completed_time = time.perf_counter()
        return BenchmarkData(
            request=json.dumps(request),
            start_time=start_time,
            completed_time=completed_time,
            query_latency=completed_time - start_time,
            first_chunk_latency=0.001,
            success=True,
            is_stream=True,
            prompt_tokens=1,
            completion_tokens=1,
            generated_text='ok',
        )


def _run_strategy(args: Arguments) -> _RecordingClient:
    client = _RecordingClient()

    async def main() -> None:
        strategy = MultiTurnStrategy(args, _ApiPlugin(), client, asyncio.Queue(), _conversations(args.total_count))
        await strategy.run()

    asyncio.run(main())
    return client


def _conv_index(conv_id: str) -> int:
    return int(conv_id.removeprefix('conv-'))


def test_measured_portion_starts_without_draining() -> None:
    parallel = 4
    args = _make_args(number=6, parallel=parallel, warmup_num=parallel)
    client = _run_strategy(args)

    measured = [event for event in client.events if _conv_index(event['conv_id']) >= parallel]
    assert measured, 'no measured conversation was dispatched'
    first_measured = min(measured, key=lambda event: event['order'])
    assert first_measured['occupancy_at_start'] == parallel - 1

    later_occupancies = [event['occupancy_at_start'] for event in client.events if event['order'] > 0]
    assert later_occupancies and all(occupancy > 0 for occupancy in later_occupancies)


def test_partial_warmup_degrades_gracefully() -> None:
    parallel = 4
    warmup = 2
    args = _make_args(number=6, parallel=parallel, warmup_num=warmup)
    client = _run_strategy(args)

    opening = sorted(client.events, key=lambda event: event['order'])[:parallel]
    assert [event['occupancy_at_start'] for event in opening] == list(range(parallel))
    assert sum(1 for event in opening if _conv_index(event['conv_id']) >= warmup) == parallel - warmup


def test_zero_warmup_keeps_all_opening_work_measured() -> None:
    parallel = 3
    args = _make_args(number=5, parallel=parallel, warmup_num=0)
    client = _run_strategy(args)

    opening = sorted(client.events, key=lambda event: event['order'])[:parallel]
    assert [event['occupancy_at_start'] for event in opening] == list(range(parallel))
    assert all(_conv_index(event['conv_id']) < args.number for event in opening)


def test_duration_exempts_warmup_and_caps_measured() -> None:
    parallel = 2
    measured = 20
    args = _make_args(number=measured, parallel=parallel, warmup_num=parallel, duration=0.0)
    client = _run_strategy(args)

    dispatched = {_conv_index(event['conv_id']) for event in client.events}
    assert {0, 1}.issubset(dispatched)
    measured_dispatched = {idx for idx in dispatched if idx >= parallel}
    assert 1 <= len(measured_dispatched) < measured


class _RecordingLogger:

    def __init__(self) -> None:
        self.warnings: List[str] = []

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def info(self, msg: str) -> None:
        pass


def test_warns_when_multi_turn_warmup_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(multi_turn_module, 'logger', recorder)

    strategy = MultiTurnStrategy(_make_args(number=10, parallel=4, warmup_num=0), _ApiPlugin(), None, asyncio.Queue(), [])
    strategy._log_warmup_handoff()

    assert len(recorder.warnings) == 1
    assert 'Multi-turn warmup is disabled' in recorder.warnings[0]
    assert '--warmup-num 4' in recorder.warnings[0]


def test_warns_when_multi_turn_warmup_smaller_than_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(multi_turn_module, 'logger', recorder)

    strategy = MultiTurnStrategy(_make_args(number=10, parallel=4, warmup_num=2), _ApiPlugin(), None, asyncio.Queue(), [])
    strategy._log_warmup_handoff()

    assert len(recorder.warnings) == 1
    assert 'covers only 2 of the 4 concurrency slots' in recorder.warnings[0]
    assert '2 measured conversation(s)' in recorder.warnings[0]


def test_silent_when_multi_turn_warmup_covers_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(multi_turn_module, 'logger', recorder)

    strategy = MultiTurnStrategy(_make_args(number=10, parallel=4, warmup_num=4), _ApiPlugin(), None, asyncio.Queue(), [])
    strategy._log_warmup_handoff()

    assert recorder.warnings == []


def test_warmup_reaches_producer_but_stays_out_of_result_db(tmp_path) -> None:
    parallel = 3
    number = 5
    output_dir = tmp_path / 'multi-turn-e2e'
    output_dir.mkdir()
    args = _make_args(number=number, parallel=parallel, warmup_num=parallel, outputs_dir=str(output_dir))
    client = _RecordingClient()
    queue: asyncio.Queue = asyncio.Queue()
    strategy = MultiTurnStrategy(args, _ApiPlugin(), client, queue, _conversations(args.total_count))

    async def main() -> None:
        await run_benchmark_pipeline(strategy.run(), queue, args, _ApiPlugin())

    asyncio.run(main())

    assert len(client.events) == parallel + number
    with sqlite3.connect(os.path.join(str(output_dir), 'benchmark_data.db')) as con:
        rows = con.execute('SELECT COUNT(*) FROM result').fetchone()[0]
    assert rows == number
