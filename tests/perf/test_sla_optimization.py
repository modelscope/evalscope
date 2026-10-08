# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for SLA auto-tune optimization (``"min"`` / ``"max"`` criteria).

A run whose requests failed reports -1 placeholders (-1000 ms TTFT), which must not be picked
as the best value.
"""
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evalscope.perf.arguments import Arguments
from evalscope.perf.sla.sla_run import SLAAutoTuner
from evalscope.perf.utils.perf_constants import Metrics


def _results(succeeded: bool, ttft_ms: float) -> dict:
    total = 4
    return {
        'metrics': {
            Metrics.TOTAL_REQUESTS: total,
            Metrics.SUCCEED_REQUESTS: total if succeeded else 0,
            Metrics.FAILED_REQUESTS: 0 if succeeded else total,
            Metrics.AVERAGE_TIME_TO_FIRST_TOKEN: ttft_ms,
        },
        'percentiles': {},
    }


def _tune(run_stub) -> dict:
    with TemporaryDirectory() as output_dir:
        args = Arguments(
            model='mock',
            outputs_dir=output_dir,
            sla_auto_tune=True,
            sla_params=[{'avg_ttft': 'min'}],
            parallel=1,
            sla_upper_bound=8,
            sla_num_runs=1,
        )
        tuner = SLAAutoTuner(args, run_stub)
        with patch('evalscope.perf.sla.sla_run.print_summary'):
            tuner.tune()
    return tuner.sla_results_table[0]


class TestSLAOptimization(unittest.TestCase):

    def test_runs_with_failed_requests_cannot_win(self):

        def run_stub(args: Arguments, _output_path: str) -> dict:
            # parallel 1 and 2 succeed; from 4 on every request fails.
            if args.parallel <= 2:
                return {'run': _results(True, 50.0 if args.parallel == 1 else 40.0)}
            return {'run': _results(False, -1000.0)}

        row = _tune(run_stub)
        self.assertEqual(row['Max Satisfied'], 2)
        self.assertIn('40', row['Note'])

    def test_no_successful_run_reports_none(self):

        def run_stub(_args: Arguments, _output_path: str) -> dict:
            return {'run': _results(False, -1000.0)}

        self.assertEqual(_tune(run_stub)['Max Satisfied'], 'None')


if __name__ == '__main__':
    unittest.main()
