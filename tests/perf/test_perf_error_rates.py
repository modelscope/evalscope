"""Status-code-derived error rates in the perf summary (#2072)."""

from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator
from evalscope.perf.utils.perf_constants import Metrics
from evalscope.perf.utils.perf_models import BenchmarkSummary


def _failed(status_code: int) -> BenchmarkData:
    return BenchmarkData(success=False, status_code=status_code)


def _accumulate(*status_codes: int) -> MetricsAccumulator:
    accumulator = MetricsAccumulator()
    for code in status_codes:
        accumulator.update(_failed(code), api_plugin=None)
    return accumulator


def test_error_classes_are_counted_separately() -> None:
    """429 (throttling) and 5xx (faults) are distinct failure classes."""
    accumulator = _accumulate(429, 429, 500, 503, 400)

    assert accumulator.n_total == 5
    assert accumulator.n_failed == 5
    assert accumulator.n_rate_limited == 2
    assert accumulator.n_server_error == 2


def test_rates_are_percentages_of_all_requests() -> None:
    accumulator = _accumulate(429, 500, 503, 400)

    assert accumulator.error_rate == 100.0
    assert accumulator.rate_limit_rate == 25.0
    assert accumulator.server_error_rate == 50.0


def test_rates_are_zero_without_requests() -> None:
    accumulator = MetricsAccumulator()

    assert accumulator.error_rate == 0.0
    assert accumulator.rate_limit_rate == 0.0
    assert accumulator.server_error_rate == 0.0


def test_summary_reports_the_rates() -> None:
    accumulator = _accumulate(429, 500, 400)
    message = accumulator.to_result().create_message()

    assert message[Metrics.ERROR_RATE] == 100.0
    assert message[Metrics.RATE_LIMIT_RATE] == 33.3333
    assert message[Metrics.SERVER_ERROR_RATE] == 33.3333

    summary = BenchmarkSummary.from_dict(message)
    assert summary.error_rate == 100.0
    assert 'Error Rate (%)' in summary.to_dict()


def test_summary_table_shows_the_rates_when_requests_failed() -> None:
    accumulator = _accumulate(429, 500, 400)
    table = BenchmarkSummary.from_dict(accumulator.to_result().create_message()).to_table()

    assert 'Error Rate (%)' in table
    assert '429 Rate (%)' in table
    assert '5xx Rate (%)' in table


def test_summary_table_hides_the_rates_without_failures() -> None:
    accumulator = MetricsAccumulator()
    table = BenchmarkSummary.from_dict(accumulator.to_result().create_message()).to_table()

    assert 'Error Rate (%)' not in table