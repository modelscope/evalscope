"""Per-device throughput rows derived from a declared --device-count.

Undeclared (the default) records nothing: no new serialized keys, no rendered
rows. Declared counts divide the aggregate throughputs; a missing aggregate
stays None instead of a fake rescale (no "-1 / N" negative throughput).
"""
import pytest
from pydantic import ValidationError

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator
from evalscope.perf.utils.perf_constants import Metrics
from evalscope.perf.utils.perf_models import BenchmarkSummary


def _ok(completion_tokens: int) -> BenchmarkData:
    data = BenchmarkData(request={})
    data.success = True
    data.start_time = 0.0
    data.completed_time = 1.0
    data.chunk_times = [0.5]
    data.prompt_tokens = 10
    data.completion_tokens = completion_tokens
    return data


def _accumulate(device_count, *requests):
    acc = MetricsAccumulator(device_count=device_count)
    for req in requests:
        acc.update(req, api_plugin=None)
    return acc


def test_undeclared_records_nothing() -> None:
    # default (no --device-count): no new keys in the message, no summary fields,
    # no rendered rows — the payload shape is unchanged
    result = _accumulate(None, _ok(400)).to_result()

    assert result.device_count is None
    assert result.output_throughput_per_device is None
    assert result.total_throughput_per_device is None

    message = result.create_message(api_type='openai')
    assert Metrics.DEVICE_COUNT not in message
    assert Metrics.OUTPUT_THROUGHPUT_PER_DEVICE not in message
    assert Metrics.TOTAL_THROUGHPUT_PER_DEVICE not in message

    table = BenchmarkSummary.from_dict(message).to_table()
    assert 'Device Count' not in table
    assert 'per Device' not in table


def test_declared_count_derives_per_device_rows() -> None:
    # 400 output + 420 total tokens over 1 s of wall time, 4 declared devices
    result = _accumulate(4, _ok(200), _ok(200)).to_result()

    assert result.avg_output_token_throughput == pytest.approx(400.0)
    assert result.avg_total_token_throughput == pytest.approx(420.0)
    assert result.output_throughput_per_device == pytest.approx(100.0)
    assert result.total_throughput_per_device == pytest.approx(105.0)

    message = result.create_message(api_type='openai')
    assert message[Metrics.DEVICE_COUNT] == 4
    assert message[Metrics.OUTPUT_THROUGHPUT_PER_DEVICE] == pytest.approx(100.0)
    assert message[Metrics.TOTAL_THROUGHPUT_PER_DEVICE] == pytest.approx(105.0)

    table = BenchmarkSummary.from_dict(message).to_table()
    assert 'Device Count' in table
    assert 'Output Throughput per Device (tok/s)' in table
    assert 'Total Throughput per Device (tok/s)' in table


def test_zero_success_reports_zero_per_device() -> None:
    # no successful requests → 0 tokens over the wall time, so the aggregate
    # throughput is a factual 0.0 and the per-device value mirrors it; the
    # >= 0 guard only excludes the -1 (missing) fallback, never a real zero
    failed = BenchmarkData(request={})
    failed.success = False
    failed.start_time = 0.0
    failed.completed_time = 1.0
    result = _accumulate(4, failed).to_result()

    assert result.avg_output_token_throughput == 0.0
    assert result.output_throughput_per_device == 0.0
    assert result.total_throughput_per_device == 0.0

    message = result.create_message(api_type='openai')
    assert message[Metrics.DEVICE_COUNT] == 4
    assert message[Metrics.OUTPUT_THROUGHPUT_PER_DEVICE] == 0.0


def test_legacy_summary_json_still_loads() -> None:
    # a persisted summary without the new keys loads with None defaults
    summary = BenchmarkSummary.model_validate({Metrics.OUTPUT_TOKEN_THROUGHPUT: 100.0})
    assert summary.device_count is None
    assert summary.output_throughput_per_device is None


def test_device_count_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Arguments(model='m', url='http://x', device_count=0)
    with pytest.raises(ValidationError):
        Arguments(model='m', url='http://x', device_count=-2)
    assert Arguments(model='m', url='http://x', device_count=8).device_count == 8
    assert Arguments(model='m', url='http://x').device_count is None
