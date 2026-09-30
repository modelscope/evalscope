# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for SLA metric units.

A threshold either carries its own unit or is read in the unit the perf reports use, so it can be
copied off a report without conversion.
"""
import unittest

from evalscope.perf.sla.sla_run import SLA_METRIC_FIELDS, check_sla, get_metric_values, parse_sla_params
from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics


def _results(ttft_ms: float, tpot_ms: float, latency_s: float = 2.0) -> dict:
    """Benchmark results in the units the perf contract reports them in."""
    return {
        'metrics': {
            Metrics.TOTAL_REQUESTS: 10,
            Metrics.SUCCEED_REQUESTS: 10,
            Metrics.FAILED_REQUESTS: 0,
            Metrics.AVERAGE_LATENCY: latency_s,
            Metrics.AVERAGE_TIME_TO_FIRST_TOKEN: ttft_ms,
            Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN: tpot_ms,
        },
        'percentiles': {
            PercentileMetrics.PERCENTILES: ['50%', '90%', '99%'],
            PercentileMetrics.LATENCY: [latency_s] * 3,
            PercentileMetrics.TTFT: [ttft_ms] * 3,
            PercentileMetrics.TPOT: [tpot_ms] * 3,
        },
    }


class TestSLAMetricUnits(unittest.TestCase):

    def test_values_keep_their_reported_units(self):
        values = get_metric_values(_results(ttft_ms=40.0, tpot_ms=20.0))

        for key in ('avg_ttft', 'p50_ttft', 'p90_ttft', 'p99_ttft'):
            self.assertAlmostEqual(values[key], 40.0)
        for key in ('avg_tpot', 'p50_tpot', 'p90_tpot', 'p99_tpot'):
            self.assertAlmostEqual(values[key], 20.0)
        self.assertAlmostEqual(values['p99_latency'], 2.0)

    def test_unit_suffix_is_converted_to_the_metric_unit(self):
        self.assertAlmostEqual(parse_sla_params('[{"avg_ttft": "<=2s"}]')[0]['avg_ttft'].target, 2000.0)
        self.assertAlmostEqual(parse_sla_params('[{"avg_latency": "<=500ms"}]')[0]['avg_latency'].target, 0.5)
        self.assertEqual(str(parse_sla_params('[{"avg_ttft": "<=2s"}]')[0]['avg_ttft']), '<= 2000 ms')

    def test_bare_number_is_read_in_the_reported_unit(self):
        results = _results(ttft_ms=40.0, tpot_ms=20.0)

        self.assertTrue(check_sla(results, parse_sla_params('[{"avg_ttft": "<=50"}]')))
        self.assertFalse(check_sla(results, parse_sla_params('[{"avg_ttft": "<=30"}]')))
        self.assertTrue(check_sla(results, parse_sla_params('[{"p99_latency": "<=2"}]')))

    def test_documented_thresholds(self):
        results = _results(ttft_ms=40.0, tpot_ms=20.0)

        self.assertTrue(check_sla(results, parse_sla_params('[{"avg_ttft": "<=2s", "avg_tpot": "<=50ms"}]')))
        self.assertTrue(check_sla(results, parse_sla_params('[{"p99_ttft": "<50ms"}]')))
        self.assertFalse(check_sla(results, parse_sla_params('[{"p99_ttft": "<10ms"}]')))

    def test_unit_of_another_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_sla_params('[{"rps": ">=5ms"}]')

    def test_unknown_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_sla_params('[{"avg_ttf": "<=2s"}]')

    def test_metric_names_cover_the_compared_values(self):
        self.assertEqual(set(SLA_METRIC_FIELDS), set(get_metric_values(_results(1.0, 1.0))))


if __name__ == '__main__':
    unittest.main()
