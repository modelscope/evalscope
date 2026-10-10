"""Regression tests for a successful-but-usage-less sample (see #1817).

A request can be classified as *successful* while carrying no token usage --
for example an OpenAI Responses SSE stream that closes after emitting only
delta events, with no ``response.completed`` / ``response.incomplete`` event
and therefore no usage block.  When no ``--tokenizer-path`` is configured the
plugin cannot recompute the counts either.

Previously ``BenchmarkData.finalize()`` let the plugin's ``ValueError`` escape.
``MetricsAccumulator.update()`` runs unguarded inside the metrics-consumer
loop, that exception propagated out of ``statistic_benchmark_metric`` and out
of ``run_benchmark_pipeline``, aborting the whole benchmark run.

Expected behaviour: such a sample is still counted as a successful request
(latency / QPS / throughput) but is *excluded* from the token averages rather
than counted as zero tokens.
"""
import asyncio
from typing import Any, List

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.metrics_consumer import statistic_benchmark_metric
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator

_NO_USAGE_MSG = (
    'Error: Unable to retrieve usage information from OpenAI Responses API response and no tokenizer was '
    'specified. Please ensure the API returns usage or set --tokenizer-path.'
)


class _NoUsagePlugin:
    """Plugin that reports success but cannot supply token usage."""

    def parse_responses(self, responses, request=None, **kwargs):
        raise ValueError(_NO_USAGE_MSG)


class _UsagePlugin:
    """Plugin that returns a fixed token usage."""

    def parse_responses(self, responses, request=None, **kwargs):
        return 10, 5


def _bench_data(index: int = 0, success: bool = True) -> BenchmarkData:
    data = BenchmarkData(
        success=success,
        is_stream=True,
        start_time=float(index),
        completed_time=float(index) + 1.0,
        query_latency=1.0,
        first_chunk_latency=0.2,
        prompt_tokens=None,
        completion_tokens=None,
    )
    data.request = '{}'
    data.response_messages = []
    return data


def _make_args(tmp_path, **kwargs: Any) -> Arguments:
    kwargs.setdefault('log_every_n_query', 100)
    args = Arguments(model='test-model', api='openai', **kwargs)
    args.number = 2
    args.outputs_dir = str(tmp_path)
    return args


def _drive_consumer(args: Arguments, records: List[BenchmarkData], plugin):
    """Feed `records` through the real metrics-consumer loop.

    Returns the finished ``BenchmarkMetrics`` snapshot (what
    ``statistic_benchmark_metric`` resolves to).
    """

    async def go():
        queue: asyncio.Queue = asyncio.Queue()
        completed = asyncio.Event()
        consumer_task = asyncio.create_task(statistic_benchmark_metric(queue, args, plugin, completed))
        for record in records:
            await queue.put(record)
        completed.set()
        result, _, _, _ = await consumer_task
        return result

    return asyncio.run(go())


class TestUsageLessSuccess:
    """#1817: a usage-less success must not abort the run."""

    def test_usage_less_success_does_not_abort_consumer(self, tmp_path) -> None:
        args = _make_args(tmp_path)

        # Would previously raise ValueError out of the metrics consumer.
        result = _drive_consumer(args, [_bench_data()], _NoUsagePlugin())

        assert result.succeed_requests == 1
        assert result.total_requests == 1
        # The run completed and is still counted as successful -> its latency
        # participates in the averages.
        assert result.avg_latency == pytest.approx(1.0)

    def test_usage_less_sample_excluded_from_token_averages(self) -> None:
        acc = MetricsAccumulator(concurrency=1)
        acc.update(_bench_data(0), _UsagePlugin())
        acc.update(_bench_data(1), _NoUsagePlugin())

        result = acc.to_result()
        assert result.succeed_requests == 2
        assert acc.n_token_success == 1
        # Exactly the usage-reporting sample contributes: not (10/2, 5/2).
        assert result.avg_prompt_tokens == pytest.approx(10.0)
        assert result.avg_completion_tokens == pytest.approx(5.0)

    def test_all_usage_less_reports_token_averages_unavailable(self) -> None:
        acc = MetricsAccumulator(concurrency=1)
        acc.update(_bench_data(0), _NoUsagePlugin())

        result = acc.to_result()
        assert result.succeed_requests == 1
        # No sample could report tokens -> "not available" rather than 0.
        assert result.avg_prompt_tokens == -1
        assert result.avg_completion_tokens == -1

    def test_usage_reporting_path_unchanged(self) -> None:
        acc = MetricsAccumulator(concurrency=1)
        acc.update(_bench_data(0), _UsagePlugin())
        acc.update(_bench_data(1), _UsagePlugin())

        result = acc.to_result()
        assert acc.n_token_success == 2
        assert result.avg_prompt_tokens == pytest.approx(10.0)
        assert result.avg_completion_tokens == pytest.approx(5.0)
        assert result.avg_latency == pytest.approx(1.0)
