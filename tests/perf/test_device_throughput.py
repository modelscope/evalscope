"""Regression coverage for device-normalized throughput reporting."""

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.benchmark_util import BenchmarkMetrics
from evalscope.perf.utils.perf_constants import Metrics
from evalscope.perf.utils.perf_models import BenchmarkSummary
from evalscope.perf.utils.report.summary import build_summary_items


def _summary(**values) -> BenchmarkSummary:
    return BenchmarkSummary.model_validate(values)


def test_device_throughput_is_aggregate_throughput_per_device():
    result = BenchmarkMetrics(
        device_count=4,
        avg_output_token_throughput=80.0,
        avg_total_token_throughput=120.0,
    )

    message = result.create_message()

    assert message[Metrics.DEVICE_COUNT] == 4
    assert message[Metrics.OUTPUT_TOKEN_THROUGHPUT_PER_DEVICE] == 20.0
    assert message[Metrics.TOTAL_TOKEN_THROUGHPUT_PER_DEVICE] == 30.0


def test_missing_device_count_omits_device_fields_from_all_summaries():
    message = BenchmarkMetrics().create_message()
    summary = _summary(**message)

    assert not {
        Metrics.DEVICE_COUNT,
        Metrics.OUTPUT_TOKEN_THROUGHPUT_PER_DEVICE,
        Metrics.TOTAL_TOKEN_THROUGHPUT_PER_DEVICE,
    } & message.keys()
    assert Metrics.DEVICE_COUNT not in summary.to_table()
    assert Metrics.OUTPUT_TOKEN_THROUGHPUT_PER_DEVICE not in summary.to_table()
    assert Metrics.TOTAL_TOKEN_THROUGHPUT_PER_DEVICE not in summary.to_table()
    item_text = str(build_summary_items(summary, is_embedding_flag=False))
    assert 'Device Count' not in item_text
    assert 'per Device' not in item_text


def test_zero_and_unavailable_throughput_are_not_misreported_per_device():
    zero_result = BenchmarkMetrics(
        device_count=4,
        total_requests=1,
        failed_requests=1,
        avg_output_token_throughput=0.0,
        avg_total_token_throughput=0.0,
    )
    zero_message = zero_result.create_message()
    assert zero_message[Metrics.OUTPUT_TOKEN_THROUGHPUT_PER_DEVICE] == 0.0
    assert zero_message[Metrics.TOTAL_TOKEN_THROUGHPUT_PER_DEVICE] == 0.0

    unavailable_result = BenchmarkMetrics(
        device_count=4,
        avg_output_token_throughput=-1,
        avg_total_token_throughput=-1,
    )
    unavailable_message = unavailable_result.create_message()
    assert Metrics.OUTPUT_TOKEN_THROUGHPUT_PER_DEVICE not in unavailable_message
    assert Metrics.TOTAL_TOKEN_THROUGHPUT_PER_DEVICE not in unavailable_message


@pytest.mark.parametrize('device_count', [0, -1])
def test_non_positive_device_count_is_rejected(device_count):
    with pytest.raises(ValueError, match='--device-count must be >= 1'):
        Arguments(device_count=device_count)


def test_legacy_summary_without_device_fields_remains_loadable():
    summary = BenchmarkSummary.model_validate({'Total Requests': 3, 'Output Throughput (tok/s)': 12.0})

    assert summary.device_count is None
    assert summary.output_token_throughput_per_device is None
    assert summary.total_token_throughput_per_device is None


def test_declared_device_count_renders_device_rows_in_summary_table():
    summary = _summary(
        **{
            Metrics.DEVICE_COUNT: 4,
            Metrics.OUTPUT_TOKEN_THROUGHPUT_PER_DEVICE: 20.0,
            Metrics.TOTAL_TOKEN_THROUGHPUT_PER_DEVICE: 30.0,
        }
    )

    table = summary.to_table()

    assert Metrics.DEVICE_COUNT in table
    assert Metrics.OUTPUT_TOKEN_THROUGHPUT_PER_DEVICE in table
    assert Metrics.TOTAL_TOKEN_THROUGHPUT_PER_DEVICE in table
