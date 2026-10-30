"""KV-cache hit rate must not be silently inflated when some successful requests
never report ``cached_tokens`` (issue #1938).

A missing field is *not* a cache miss, so such requests stay out of the hit-rate
denominator — but the result now carries request/token coverage so readers can
tell how much of the run the hit rate actually describes.
"""

import logging
from unittest.mock import MagicMock

import pytest

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
    def test_warns_when_coverage_is_partial(self, caplog):
        acc = _accumulate(_request(100, 80), _request(100, None))
        with caplog.at_level(logging.WARNING):
            acc.to_result()

        messages = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING]
        assert any('1/2' in m and 'cached_tokens' in m for m in messages), messages

    def test_no_warning_when_coverage_is_full(self, caplog):
        acc = _accumulate(_request(100, 80), _request(100, 0))
        with caplog.at_level(logging.WARNING):
            acc.to_result()

        assert not [rec for rec in caplog.records if 'cached_tokens' in rec.getMessage()]

    def test_coverage_is_emitted_next_to_hit_rate(self):
        result = _accumulate(_request(100, 80), _request(100, None)).to_result().create_message()

        assert result[Metrics.AVERAGE_CACHED_PERCENT] == pytest.approx(80.0)
        assert result[Metrics.CACHE_REQUEST_COVERAGE_PERCENT] == pytest.approx(50.0)
        assert result[Metrics.CACHE_TOKEN_COVERAGE_PERCENT] == pytest.approx(50.0)

    def test_metrics_dataclass_defaults_to_not_applicable(self):
        metrics = BenchmarkMetrics()
        assert metrics.cache_request_coverage_percent == -1
        assert metrics.cache_token_coverage_percent == -1
