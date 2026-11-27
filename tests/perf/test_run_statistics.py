"""Run-to-run statistics for repeated performance measurements.

The statistics back the SLA auto-tuner's repeated runs (`--sla-num-runs`):
each pressure level already runs N times, and this layer reports how much the
per-run metrics move instead of discarding the spread.
"""
import math
import statistics as stdlib_statistics

import pytest

from evalscope.perf.sla.sla_models import SLAProbe
from evalscope.perf.utils.statistics import RunStatistics, summarize_runs, t_critical


def test_empty_series_has_no_summary():
    assert summarize_runs([]) is None


def test_single_run_reports_only_the_mean():
    # n = 1: no spread to estimate, and no interval either — never fabricate one
    stats = summarize_runs([1240.5])

    assert stats == RunStatistics(n=1, mean=1240.5)
    assert stats.std is None
    assert stats.ci_low is None
    assert stats.ci_half_width is None


def test_two_runs_report_std_but_no_interval():
    stats = summarize_runs([100.0, 104.0])

    assert stats.n == 2
    assert stats.mean == pytest.approx(102.0)
    assert stats.std == pytest.approx(stdlib_statistics.stdev([100.0, 104.0]))
    assert stats.ci_low is None and stats.ci_high is None
    assert stats.ci_half_width is None


def test_three_runs_report_mean_std_and_interval():
    values = [100.0, 104.0, 98.0]
    stats = summarize_runs(values)

    expected_mean = sum(values) / 3
    expected_std = stdlib_statistics.stdev(values)
    half_width = t_critical(2) * expected_std / math.sqrt(3)

    assert stats.mean == pytest.approx(expected_mean)
    assert stats.std == pytest.approx(expected_std)
    assert stats.ci_low == pytest.approx(expected_mean - half_width)
    assert stats.ci_high == pytest.approx(expected_mean + half_width)
    assert stats.ci_half_width == pytest.approx(half_width)


def test_interval_coverage_is_approximately_nominal():
    """A 95% interval should contain the true mean ~95% of the time.

    Draw 1,000 samples of n = 20 from a known normal distribution and count
    how often the interval covers the population mean. Wide tolerance keeps the
    test stable across seeds while still catching a broken critical value.
    """
    import random

    rng = random.Random(20261123)
    true_mean, true_std = 1240.0, 25.0
    n, trials = 20, 1000

    covered = 0
    for _ in range(trials):
        sample = [rng.gauss(true_mean, true_std) for _ in range(n)]
        stats = summarize_runs(sample)
        if stats.ci_low <= true_mean <= stats.ci_high:
            covered += 1

    coverage = covered / trials
    assert 0.93 <= coverage <= 0.97, f'coverage {coverage:.3f} outside [0.93, 0.97]'


def test_t_critical_table_and_normal_fallback():
    assert t_critical(1) == pytest.approx(12.706)
    assert t_critical(2) == pytest.approx(4.303)
    assert t_critical(30) == pytest.approx(2.042)
    # Beyond the table the normal approximation is used.
    assert t_critical(31) == 1.96
    assert t_critical(1000) == 1.96
    with pytest.raises(ValueError):
        t_critical(0)


def test_interval_half_width_shrinks_with_more_runs():
    """More runs of the same distribution give a tighter interval."""
    import random

    rng = random.Random(7)
    base = [rng.gauss(1000.0, 20.0) for _ in range(64)]

    assert summarize_runs(base).ci_half_width < summarize_runs(base[:5]).ci_half_width


def test_zero_variance_series_has_zero_width_interval():
    stats = summarize_runs([100.0, 100.0, 100.0, 100.0])
    assert stats.std == 0.0
    assert stats.ci_low == stats.ci_high == 100.0
    assert stats.ci_half_width == 0.0


def test_format_renders_gracefully_across_sample_sizes():
    one = summarize_runs([42.0])
    assert one.format() == '42 (n=1)'

    two = summarize_runs([100.0, 104.0])
    assert '±' in two.format() and 'CI' not in two.format()

    three = summarize_runs([100.0, 104.0, 98.0])
    rendered = three.format()
    assert '±' in rendered and '95% CI' in rendered and 'n=3' in rendered


def test_probe_carries_the_statistics_map():
    probe = SLAProbe(
        value=16,
        averaged_result={},
        total_requests=100,
        succeeded_requests=100,
        success_rate=100.0,
        request_gate_passed=True,
        valid=True,
        statistics={'tps': summarize_runs([100.0, 104.0, 98.0])},
    )

    assert probe.statistics['tps'].n == 3
    # pydantic serializes the nested dataclass, so the summary survives a dump.
    dumped = probe.model_dump()
    assert dumped['statistics']['tps']['mean'] == pytest.approx(sum([100.0, 104.0, 98.0]) / 3)
