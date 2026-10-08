import json
import pickle
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.sla.sla_run import (
    SLAResult,
    check_sla,
    format_sla_tables,
    get_metric_values,
    parse_sla_params,
    run_sla_auto_tune,
)
from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics
from evalscope.perf.utils.perf_models import BenchmarkSummary, PercentileResult


def _result(value: float, succeeded: int = 4, total: int = 4, percentile: bool = True) -> dict:
    result = {
        'metrics': {
            Metrics.TOTAL_REQUESTS: total,
            Metrics.SUCCEED_REQUESTS: succeeded,
            Metrics.FAILED_REQUESTS: total - succeeded,
            Metrics.AVERAGE_LATENCY: value,
            Metrics.OUTPUT_TOKEN_THROUGHPUT: value,
        },
        'percentiles': {PercentileMetrics.PERCENTILES: ['99%'], PercentileMetrics.LATENCY: [value]},
    }
    if percentile:
        result['percentiles'][PercentileMetrics.TTFT] = [value]
    return result


def _tune(values: dict, criteria: list, start: int = 1, runs: int = 1, api: str = 'openai') -> SLAResult:
    with TemporaryDirectory() as output_dir:
        args = Arguments(
            model='mock',
            api=api,
            outputs_dir=output_dir,
            sla_auto_tune=True,
            sla_params=criteria,
            sla_lower_bound=min(values),
            sla_upper_bound=max(values),
            sla_num_runs=runs,
            sleep_interval=0,
            parallel=start,
        )

        def runner(run_args: Arguments, _output_path: str) -> dict:
            result = values[run_args.parallel]
            return {'run': result() if callable(result) else result}

        result = run_sla_auto_tune(args, runner)
        with open(f'{output_dir}/sla_summary.json', encoding='utf-8') as summary_file:
            on_disk = json.load(summary_file)
        assert dict(result) == on_disk
        assert all(set(value) == {'metrics', 'percentiles'} for value in on_disk.values())
        return result


def test_missing_percentile_fails_without_turning_into_zero() -> None:
    missing = _result(0.0, percentile=False)
    assert 'p99_ttft' not in get_metric_values(missing)
    assert not check_sla(missing, parse_sla_params([{'p99_ttft': '<=10ms'}]))
    assert get_metric_values(_result(0.0))['p99_ttft'] == 0.0
    assert check_sla(_result(0.0), parse_sla_params([{'p99_ttft': '<=10ms'}]))


def test_one_missing_run_fails_the_pressure_point_and_keeps_exact_counts() -> None:
    attempts = iter((_result(0.04), _result(0.04, succeeded=3), _result(0.04, percentile=False)))
    result = _tune({1: lambda: next(attempts)}, [{'p99_ttft': '<=50ms'}], runs=3)
    probe = result.probes[0]
    assert (probe.succeeded_requests, probe.total_requests) == (11, 12)
    assert not probe.valid
    assert 'run 3: missing or non-finite p99_ttft' in probe.reasons
    assert result.selections[0].selected_value is None
    assert '11/12' in format_sla_tables(result)


@pytest.mark.parametrize('criterion', [{'tps': 'max'}, {'avg_latency': '<=1s'}])
def test_missing_request_counts_cannot_pass_on_other_runs_counts(criterion) -> None:
    missing_counts = _result(0.04)
    del missing_counts['metrics'][Metrics.TOTAL_REQUESTS]
    attempts = iter((_result(0.04), missing_counts))
    result = _tune({1: lambda: next(attempts)}, [criterion], runs=2)
    probe = result.probes[0]
    assert (probe.succeeded_requests, probe.total_requests) == (4, 4)
    assert not probe.request_gate_passed
    assert not probe.valid
    assert result.selections[0].selected_value is None


def test_optimization_searches_below_a_failed_start() -> None:
    values = {n: _result(40.0 if n == 1 else 50.0, succeeded=4 if n <= 2 else 0) for n in range(1, 9)}
    result = _tune(values, [{'tps': 'max'}], start=4)
    assert result.selections[0].selected_value == 2
    assert {probe.value for probe in result.probes} >= {1, 2, 4}


def test_optimization_finds_the_single_peak_and_preserves_probe_order() -> None:
    throughputs = (1, 2, 2.5, 3, 5, 4, 3, 2)
    result = _tune({n: _result(throughputs[n - 1]) for n in range(1, 9)}, [{'tps': 'max'}])
    assert result.selections[0].selected_value == 5
    assert result.selections[0].observed_metric == 5
    assert result.selections[0].status == 'best_observed'
    assert len({probe.value for probe in result.probes}) == len(result.probes)


def test_optimization_can_search_left_of_a_valid_start() -> None:
    values = (1, 5, 4, 3, 2, 1)
    result = _tune({n: _result(values[n - 1]) for n in range(1, 7)}, [{'tps': 'max'}], start=4)
    assert result.selections[0].selected_value == 2


def test_optimization_follows_a_plateau_to_a_later_peak() -> None:
    values = (1, 1, 1, 2, 3)
    result = _tune({n: _result(value) for n, value in enumerate(values, 1)}, [{'tps': 'max'}], start=2)
    assert result.selections[0].selected_value == 5


def test_optimization_checks_a_skipped_interval_between_equal_scores() -> None:
    values = (1, 2, 3, 10) + (9,) * 16
    result = _tune({n: _result(value) for n, value in enumerate(values, 1)}, [{'tps': 'max'}])
    assert result.selections[0].selected_value == 4


def test_optimization_checks_both_sides_of_a_binary_search_tie() -> None:
    values = (1, 2) + (3,) * 11 + (10, 9, 5, 2)
    result = _tune({n: _result(value) for n, value in enumerate(values, 1)}, [{'tps': 'max'}])
    assert result.selections[0].selected_value == 14


def test_minimum_search_finds_an_interior_value() -> None:
    values = (8, 6, 4, 2, 3, 5, 7, 9)
    result = _tune({n: _result(values[n - 1]) for n in range(1, 9)}, [{'avg_latency': 'min'}])
    assert result.selections[0].selected_value == 4


def test_typed_perf_results_do_not_supply_defaulted_missing_metrics() -> None:
    raw = _result(0.0, percentile=False)
    typed = {
        'metrics': BenchmarkSummary.from_dict(raw['metrics']),
        'percentiles': PercentileResult.from_transposed(raw['percentiles']),
    }
    assert 'p99_ttft' not in get_metric_values(typed)
    assert get_metric_values(typed)['avg_latency'] == 0.0


def test_each_constraint_group_has_a_machine_readable_selection() -> None:
    result = _tune(
        {n: _result(n * 0.01) for n in range(1, 5)},
        [{'avg_latency': '<=0.02s'}, {'avg_latency': '<=0.04s'}],
    )
    assert [selection.selected_value for selection in result.selections] == [2, 4]
    assert all(selection.status == 'best_observed' for selection in result.selections)
    assert len(result.probes) == 4


def test_missing_metric_does_not_invalidate_an_independent_group() -> None:
    result = _tune(
        {1: _result(0.01, percentile=False)},
        [{'p99_ttft': '<=50ms'}, {'avg_latency': '<=0.02s'}],
    )
    assert [selection.selected_value for selection in result.selections] == [None, 1]
    assert result.probes[0].group_passes == {'1': False, '2': True}


@pytest.mark.parametrize(
    'params',
    [[], [{}], [1], [{'tps': 'max', 'avg_latency': '<=1s'}], [{'tps': 'max'}, {'rps': '<=1'}]],
)
def test_empty_or_mixed_extremum_conditions_are_rejected(params) -> None:
    with pytest.raises(ValueError):
        parse_sla_params(params)


@pytest.mark.parametrize(
    'settings',
    [
        {'sla_num_runs': 0},
        {'sla_lower_bound': 9, 'sla_upper_bound': 8},
        {'sla_fixed_parallel': 0},
        {'sla_number_multiplier': 0},
        {'sla_number_multiplier': float('inf')},
    ],
)
def test_invalid_sla_settings_fail_before_running(settings) -> None:
    with pytest.raises(ValueError):
        Arguments(model='mock', sla_auto_tune=True, sla_params=[{'rps': '<=1'}], **settings)


def test_embedding_unsupported_metric_fails_before_runner() -> None:
    with TemporaryDirectory() as output_dir:
        args = Arguments(
            model='mock',
            api='embedding',
            outputs_dir=output_dir,
            sla_auto_tune=True,
            sla_params=[{'p99_ttft': '<=10ms'}],
        )
        with pytest.raises(ValueError, match='Metrics unavailable'):
            run_sla_auto_tune(args, lambda *_: pytest.fail('runner must not be called'))


def test_perf_entrypoint_preserves_pressure_keyed_result() -> None:
    with TemporaryDirectory() as output_dir:
        args = Arguments(
            model='mock',
            outputs_dir=output_dir,
            no_timestamp=True,
            sla_auto_tune=True,
            sla_params=[{'avg_latency': '<=0.02s'}],
            sla_lower_bound=1,
            sla_upper_bound=2,
            sla_num_runs=1,
        )

        def runner(run_args: Arguments, _output_path: str) -> dict:
            return {'run': _result(run_args.parallel * 0.01)}

        with patch('evalscope.perf.main.run_one_benchmark', side_effect=runner):
            result = run_perf_benchmark(args)
        assert isinstance(result, dict)
        assert result.selections[0].selected_value == 2
        with open(f'{output_dir}/mock/sla_summary.json', encoding='utf-8') as summary_file:
            assert json.load(summary_file) == dict(result)
        assert all(key.startswith('parallel_') for key in result)
        assert all('metrics' in value for value in result.values())
        restored = pickle.loads(pickle.dumps(result))
        assert restored == result
        assert restored.selections[0].selected_value == 2


def test_rate_search_keeps_fixed_parallel_and_request_multiplier() -> None:
    with TemporaryDirectory() as output_dir:
        args = Arguments(
            model='mock',
            outputs_dir=output_dir,
            sla_auto_tune=True,
            sla_variable='rate',
            sla_params=[{'avg_latency': '<=0.03s'}],
            rate=2,
            sla_lower_bound=1,
            sla_upper_bound=4,
            sla_fixed_parallel=7,
            sla_num_runs=1,
        )

        def runner(run_args: Arguments, _output_path: str) -> dict:
            assert run_args.parallel == 7
            assert run_args.number == 2 * run_args.rate
            return {'run': _result(run_args.rate * 0.01)}

        result = run_sla_auto_tune(args, runner)
        assert all(key.startswith('rate_') for key in result)
        assert result.selections[0].selected_value == 3
