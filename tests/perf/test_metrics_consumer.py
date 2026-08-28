"""Unit tests for the perf metrics consumer's visualizer fast path.

The consumer used to build a metrics snapshot
(``accumulator.to_result().create_message(...)``) and jump to a worker thread
(``await asyncio.to_thread(maybe_log_to_visualizer, ...)``) for *every*
request even when ``args.visualizer`` is None (the default), where
``maybe_log_to_visualizer`` is a no-op.  The snapshot+thread hop must only
happen when a visualizer is actually configured.
"""
import asyncio
from typing import Any, List

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.core import metrics_consumer
from evalscope.perf.core.metrics_consumer import statistic_benchmark_metric
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator


class _DummyPlugin:
    """Minimal api plugin: finalize() only needs parse_responses()."""

    def parse_responses(self, responses, request=None, **kwargs):
        return 10, 5


def _bench_data(index: int) -> BenchmarkData:
    data = BenchmarkData(
        success=True,
        start_time=float(index),
        completed_time=float(index) + 1.0,
        query_latency=1.0,
        first_chunk_latency=0.2,
        prompt_tokens=10,
        completion_tokens=5,
        is_stream=True,
    )
    data.request = '{}'
    data.response_messages = []
    return data


def _run_consumer(args: Arguments, n_requests: int) -> MetricsAccumulator:
    """Drive statistic_benchmark_metric with n_requests successful records."""

    async def go() -> MetricsAccumulator:
        queue: asyncio.Queue = asyncio.Queue()
        completed = asyncio.Event()
        consumer_task = asyncio.create_task(statistic_benchmark_metric(queue, args, _DummyPlugin(), completed))
        for i in range(n_requests):
            await queue.put(_bench_data(i))
        completed.set()
        accumulator, trace_summary, timeline, db_path = await consumer_task
        return accumulator

    return asyncio.run(go())


def _make_args(tmp_path, **kwargs: Any) -> Arguments:
    kwargs.setdefault('log_every_n_query', 100)
    args = Arguments(model='test-model', api='openai', **kwargs)
    args.number = 2
    args.outputs_dir = str(tmp_path)
    return args


@pytest.fixture
def counters(monkeypatch: pytest.MonkeyPatch):
    """Record to_result() and maybe_log_to_visualizer() invocations."""
    calls = {'to_result': 0}
    original_to_result = MetricsAccumulator.to_result

    def counting_to_result(self: MetricsAccumulator):
        calls['to_result'] += 1
        return original_to_result(self)

    recorded_visualizer_calls: List[dict] = []

    def fake_visualizer(args: Arguments, message: dict) -> None:
        recorded_visualizer_calls.append(message)

    monkeypatch.setattr(MetricsAccumulator, 'to_result', counting_to_result)
    monkeypatch.setattr(metrics_consumer, 'maybe_log_to_visualizer', fake_visualizer)
    return calls, recorded_visualizer_calls


class TestVisualizerFastPath:

    def test_no_visualizer_skips_per_request_snapshot(self, tmp_path, counters) -> None:
        calls, visualizer_calls = counters
        args = _make_args(tmp_path)  # visualizer=None (default)

        accumulator = _run_consumer(args, n_requests=2)

        assert visualizer_calls == []
        # Only the final result snapshot; none of the former per-request ones.
        assert calls['to_result'] == 1
        assert accumulator.succeed_requests == 2

    def test_visualizer_configured_logs_per_request(self, tmp_path, counters) -> None:
        calls, visualizer_calls = counters
        args = _make_args(tmp_path, visualizer='swanlab')

        accumulator = _run_consumer(args, n_requests=2)

        assert len(visualizer_calls) == 2
        # 2 per-request snapshots + 1 final.
        assert calls['to_result'] == 3
        assert accumulator.succeed_requests == 2

    def test_periodic_logging_still_emits_without_visualizer(self, tmp_path, counters, monkeypatch) -> None:
        logged: List[str] = []
        monkeypatch.setattr(metrics_consumer.logger, 'info', lambda msg, *a, **kw: logged.append(str(msg)))
        args = _make_args(tmp_path, log_every_n_query=1)

        accumulator = _run_consumer(args, n_requests=2)

        assert accumulator.succeed_requests == 2
        # Every request crossed the log_every_n_query boundary and logged its
        # metrics message even though no visualizer is configured.  Other
        # logger.info traffic (db path, progress) is filtered out.
        metric_messages = [msg for msg in logged if msg.startswith('{')]
        assert len(metric_messages) == 2
        assert all('"Success Requests"' in msg for msg in metric_messages)
