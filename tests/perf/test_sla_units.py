# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for SLA metric units.

TTFT and TPOT are reported in milliseconds, while SLA thresholds are given in
seconds (docs/en/user_guides/stress_test/sla_auto_tune.md).
"""
import unittest

from evalscope.perf.sla.sla_run import check_sla, get_metric_values, parse_sla_params
from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics


def _results(ttft_ms: float, tpot_ms: float) -> dict:
    """Build benchmark results in the reported units: TTFT and TPOT in ms, latency in s."""
    return {
        'metrics': {
            Metrics.TOTAL_REQUESTS: 10,
            Metrics.SUCCEED_REQUESTS: 10,
            Metrics.FAILED_REQUESTS: 0,
            Metrics.AVERAGE_LATENCY: 2.0,
            Metrics.AVERAGE_TIME_TO_FIRST_TOKEN: ttft_ms,
            Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN: tpot_ms,
        },
        'percentiles': {
            PercentileMetrics.PERCENTILES: ['50%', '90%', '99%'],
            PercentileMetrics.LATENCY: [2.0, 2.0, 2.0],
            PercentileMetrics.TTFT: [ttft_ms] * 3,
            PercentileMetrics.TPOT: [tpot_ms] * 3,
        },
    }


class TestSLACheckUnits(unittest.TestCase):

    def test_ttft_and_tpot_are_compared_in_seconds(self):
        results = _results(ttft_ms=40.0, tpot_ms=20.0)

        values = get_metric_values(results)
        for key in ('avg_ttft', 'p50_ttft', 'p90_ttft', 'p99_ttft'):
            self.assertAlmostEqual(values[key], 0.04)
        for key in ('avg_tpot', 'p50_tpot', 'p90_tpot', 'p99_tpot'):
            self.assertAlmostEqual(values[key], 0.02)
        self.assertAlmostEqual(values['p99_latency'], 2.0)

        # Examples from the SLA auto-tune docs
        self.assertTrue(check_sla(results, parse_sla_params('[{"avg_ttft": "<=2", "avg_tpot": "<=0.05"}]')))
        self.assertTrue(check_sla(results, parse_sla_params('[{"p99_ttft": "<0.05"}]')))
        self.assertFalse(check_sla(results, parse_sla_params('[{"p99_ttft": "<0.01"}]')))


if __name__ == '__main__':
    unittest.main()
