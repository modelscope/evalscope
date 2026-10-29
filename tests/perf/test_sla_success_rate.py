# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the SLA 100% success-rate gate.

The gate must fail whenever any request failed, including after several runs are averaged
into fractional counts and when the rate would round to 100.0.
"""
import unittest

from evalscope.perf.sla.sla_run import check_sla
from evalscope.perf.utils.db_util import average_results
from evalscope.perf.utils.perf_constants import Metrics


def _metrics(total: float, succeed: float) -> dict:
    return {
        Metrics.TOTAL_REQUESTS: total,
        Metrics.SUCCEED_REQUESTS: succeed,
        Metrics.FAILED_REQUESTS: total - succeed,
    }


class TestSLASuccessRate(unittest.TestCase):

    def test_a_failure_in_one_averaged_run_fails_the_gate(self):
        # sla_num_runs defaults to 3; one of the runs lost a request.
        runs = [{'metrics': _metrics(8, succeed), 'percentiles': {}} for succeed in (8, 8, 7)]
        self.assertFalse(check_sla(average_results(runs), []))

    def test_a_rate_that_rounds_to_100_fails_the_gate(self):
        self.assertFalse(check_sla({'metrics': _metrics(10000, 9999)}, []))

    def test_all_requests_succeeding_passes_the_gate(self):
        runs = [{'metrics': _metrics(8, 8), 'percentiles': {}} for _ in range(3)]
        self.assertTrue(check_sla(average_results(runs), []))


if __name__ == '__main__':
    unittest.main()
