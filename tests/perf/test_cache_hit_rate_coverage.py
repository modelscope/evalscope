"""KV-cache hit rate must not be silently inflated when some successful requests
never report ``cached_tokens`` (issue #1938).

A missing field is *not* a cache miss, so such requests stay out of the hit-rate
denominator — but the result now carries request/token coverage so readers can
tell how much of the run the hit rate actually describes.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies.multi_turn import MultiTurnStrategy
from evalscope.perf.plugin.datasets.base import Turn
from evalscope.perf.utils.benchmark_util import BenchmarkData, BenchmarkMetrics, MetricsAccumulator
from evalscope.perf.utils.perf_constants import Metrics


def _request(prompt_tokens: int, cached_tokens):
    """A finalized successful request; ``cached_tokens=None`` means the server omitted the field."""
    data = BenchmarkData(request={})
    data.success = True
    data.start_time = 0.0
    data.completed_time = 1.0
    data.chunk_times = [0.5]
    data.prompt_tokens = prompt_tokens
    data.completion_tokens = 10
    data.cached_tokens = cached_tokens
    return data


def _plugin():
    plugin = MagicMock()
    plugin.parse_responses.return_value = (0, 0)
    return plugin


def _accumulate(*requests):
    acc = MetricsAccumulator()
    plugin = _plugin()
    for req in requests:
        acc.update(req, plugin)
    return acc


class TestHitRateCoverage:
    def test_example_from_issue_reports_partial_coverage(self):
        # A: 100 input tokens, 80 cached.  B: 100 input tokens, field omitted.
        result = _accumulate(_request(100, 80), _request(100, None)).to_result()

        # Hit rate is still computed over the reporting sample only (80/100),
        # NOT force-zeroed to 80/200 — a missing field is not a known miss.
        assert result.avg_cached_percent == pytest.approx(80.0)
        # ...but coverage now exposes that only half the run is represented.
        assert result.cache_request_coverage_percent == pytest.approx(50.0)
        assert result.cache_token_coverage_percent == pytest.approx(50.0)

    def test_full_coverage_when_every_request_reports(self):
        result = _accumulate(_request(100, 80), _request(100, 0)).to_result()

        assert result.avg_cached_percent == pytest.approx(40.0)
        assert result.cache_request_coverage_percent == pytest.approx(100.0)
        assert result.cache_token_coverage_percent == pytest.approx(100.0)

    def test_explicit_zero_counts_as_reported(self):
        result = _accumulate(_request(100, 0)).to_result()

        assert result.avg_cached_percent == pytest.approx(0.0)
        assert result.cache_request_coverage_percent == pytest.approx(100.0)

    def test_token_coverage_weights_by_prompt_size(self):
        # Reporting request is small, silent request is large → token coverage < request coverage.
        result = _accumulate(_request(10, 5), _request(90, None)).to_result()

        assert result.cache_request_coverage_percent == pytest.approx(50.0)
        assert result.cache_token_coverage_percent == pytest.approx(10.0)

    def test_not_applicable_when_nothing_reports(self):
        result = _accumulate(_request(100, None), _request(50, None)).to_result()

        assert result.avg_cached_percent == -1
        assert result.cache_request_coverage_percent == -1
        assert result.cache_token_coverage_percent == -1
        assert Metrics.AVERAGE_CACHED_PERCENT not in result.create_message()
        assert Metrics.CACHE_REQUEST_COVERAGE_PERCENT not in result.create_message()

    def test_failed_requests_do_not_count_toward_coverage(self):
        failed = _request(100, None)
        failed.success = False
        result = _accumulate(_request(100, 80), failed).to_result()

        assert result.cache_request_coverage_percent == pytest.approx(100.0)
        assert result.cache_token_coverage_percent == pytest.approx(100.0)


class TestWarningAndSerialization:
    def test_final_warning_when_coverage_is_partial(self, monkeypatch):
        warning = MagicMock()
        monkeypatch.setattr('evalscope.perf.utils.benchmark_util.logger.warning', warning)
        acc = _accumulate(_request(100, 80), _request(100, None))

        acc.to_result()
        acc.to_result()
        warning.assert_not_called()

        acc.log_cache_coverage_warning()
        warning.assert_called_once()
        message = warning.call_args.args[0]
        assert '1/2' in message
        assert 'cached_tokens' in message

    def test_no_warning_when_coverage_is_full(self, monkeypatch):
        warning = MagicMock()
        monkeypatch.setattr('evalscope.perf.utils.benchmark_util.logger.warning', warning)
        acc = _accumulate(_request(100, 80), _request(100, 0))

        acc.log_cache_coverage_warning()

        warning.assert_not_called()

    def test_final_warning_when_nothing_reports(self, monkeypatch):
        warning = MagicMock()
        monkeypatch.setattr('evalscope.perf.utils.benchmark_util.logger.warning', warning)
        acc = _accumulate(_request(100, None), _request(50, None))

        acc.log_cache_coverage_warning()

        warning.assert_called_once()
        message = warning.call_args.args[0]
        assert '0/2' in message
        assert 'unavailable' in message

    def test_coverage_is_emitted_next_to_hit_rate(self):
        result = _accumulate(_request(100, 80), _request(100, None)).to_result().create_message()

        assert result[Metrics.AVERAGE_CACHED_PERCENT] == pytest.approx(80.0)
        assert result[Metrics.CACHE_REQUEST_COVERAGE_PERCENT] == pytest.approx(50.0)
        assert result[Metrics.CACHE_TOKEN_COVERAGE_PERCENT] == pytest.approx(50.0)

    def test_metrics_dataclass_defaults_to_not_applicable(self):
        metrics = BenchmarkMetrics()
        assert metrics.cache_request_coverage_percent == -1
        assert metrics.cache_token_coverage_percent == -1


class _MultiTurnCacheClient:
    def __init__(self, reported: list[int | None]) -> None:
        self._reported = reported
        self._calls = 0

    async def post(self, request: dict) -> BenchmarkData:
        real_cached_tokens = self._reported[self._calls]
        self._calls += 1
        return BenchmarkData(
            request={},
            success=True,
            start_time=float(self._calls),
            completed_time=float(self._calls) + 1.0,
            query_latency=1.0,
            prompt_tokens=100,
            completion_tokens=10,
            generated_text='ok',
            real_cached_tokens=real_cached_tokens,
        )


def _run_multi_turn(reported: list[int | None]) -> tuple[MetricsAccumulator, list[BenchmarkData]]:
    args = Arguments(model='test-model', api='openai', number=1, parallel=1, rate=-1, multi_turn=True)
    args.number = 1
    args.parallel = 1
    args.rate = -1
    plugin = _plugin()
    client = _MultiTurnCacheClient(reported)
    queue: asyncio.Queue = asyncio.Queue()
    conversations = [[
        Turn(messages=[{'role': 'user', 'content': 'first'}]),
        Turn(messages=[{'role': 'user', 'content': 'second'}]),
    ]]
    strategy = MultiTurnStrategy(args, plugin, client, queue, conversations)
    asyncio.run(strategy.run())

    turns = [queue.get_nowait(), queue.get_nowait()]
    accumulator = MetricsAccumulator()
    for turn in turns:
        accumulator.update(turn, plugin)
    return accumulator, turns


def test_multi_turn_missing_reports_do_not_create_estimated_hits() -> None:
    accumulator, turns = _run_multi_turn([None, None])

    result = accumulator.to_result()

    assert [turn.cached_tokens for turn in turns] == [None, None]
    assert result.avg_cached_percent == -1
    assert result.cache_request_coverage_percent == -1
    assert result.cache_token_coverage_percent == -1


def test_multi_turn_mixed_estimates_and_reports_use_only_reports() -> None:
    accumulator, turns = _run_multi_turn([None, 50])

    result = accumulator.to_result()

    assert [turn.cached_tokens for turn in turns] == [None, 50]
    assert result.avg_cached_percent == pytest.approx(50.0)
    assert result.cache_request_coverage_percent == pytest.approx(50.0)
    assert result.cache_token_coverage_percent == pytest.approx(50.0)
