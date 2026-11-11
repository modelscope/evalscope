"""Service-reliability rates must classify failures the way the performance
review taxonomy does: 429 rate-limited, 5xx server faults, client timeouts,
and connection failures are counted separately (shares of total requests),
because their remediation differs.

A server answer (429/5xx) is classified by ``status_code`` first; a request
that never got a response is classified by the transport-reported
``failure_kind`` (timeout / connection / client error) — a rate-limited
request is therefore never double-counted as a timeout.
"""
import pytest

from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator
from evalscope.perf.utils.perf_constants import Metrics


def _failed(status_code=None, failure_kind=None):
    data = BenchmarkData(request={})
    data.success = False
    data.start_time = 0.0
    data.completed_time = 1.0
    data.status_code = status_code
    data.failure_kind = failure_kind
    data.error = 'failure'
    return data


def _ok():
    data = BenchmarkData(request={})
    data.success = True
    data.start_time = 0.0
    data.completed_time = 1.0
    data.chunk_times = [0.5]
    data.prompt_tokens = 10
    data.completion_tokens = 10
    return data


def _accumulate(*requests):
    acc = MetricsAccumulator()
    plugin = type('P', (), {'parse_responses': staticmethod(lambda *a, **k: (0, 0))})()
    for req in requests:
        acc.update(req, plugin)
    return acc


def test_rates_classified_by_status_first():
    result = _accumulate(
        _ok(),
        _failed(status_code=429),
        _failed(status_code=429),
        _failed(status_code=500),
        _failed(status_code=503),
    ).to_result()

    assert result.total_requests == 5
    assert result.error_rate == pytest.approx(80.0)  # 4 failed / 5 total
    assert result.rate_limited_rate == pytest.approx(40.0)  # 2 x 429
    assert result.server_error_rate == pytest.approx(40.0)  # 2 x 5xx
    assert result.timeout_rate == pytest.approx(0.0)
    assert result.disconnect_rate == pytest.approx(0.0)


def test_transport_failures_use_failure_kind():
    result = _accumulate(
        _ok(),
        _failed(failure_kind='timeout'),
        _failed(failure_kind='connection'),
        _failed(failure_kind='client_error'),
    ).to_result()

    assert result.error_rate == pytest.approx(75.0)
    assert result.timeout_rate == pytest.approx(25.0)
    assert result.disconnect_rate == pytest.approx(25.0)
    assert result.rate_limited_rate == pytest.approx(0.0)
    assert result.server_error_rate == pytest.approx(0.0)
    # client errors are failures but have no dedicated rate in this taxonomy;
    # they remain visible through error_rate - timeout - 429 - 5xx - disconnect
    assert result.error_rate - (
        result.timeout_rate + result.rate_limited_rate + result.server_error_rate + result.disconnect_rate
    ) == pytest.approx(25.0)


def test_status_code_wins_over_failure_kind():
    # a rate-limited answer must not also be counted as a client error
    result = _accumulate(_failed(status_code=429, failure_kind='client_error')).to_result()
    assert result.rate_limited_rate == pytest.approx(100.0)
    assert result.timeout_rate == 0.0 and result.disconnect_rate == 0.0


def test_no_requests_means_not_applicable():
    result = _accumulate().to_result()
    assert result.error_rate == -1 and result.timeout_rate == -1


def test_constants_exist():
    assert Metrics.ERROR_RATE == 'Error Rate (%)'
    assert Metrics.RATE_LIMITED_RATE == '429 Rate (%)'
    assert Metrics.SERVER_ERROR_RATE == 'Server Error Rate (5xx, %)'
