# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for perf HTML-report run discovery and labeling.

Network-free: builds minimal fake run sub-directories on disk and exercises
``RunLoader.load_all`` plus the ``RunData``/chart helpers directly.

Regression for open-loop runs with no fixed rate (``rate_-1.0_number_<M>``),
which were previously dropped by the loader regex so no HTML report was
generated.
"""
import json
import os
import tempfile
import unittest


def _write_json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def _make_run(base: str, dir_name: str, *, request_rate: float) -> None:
    """Create a minimal run sub-directory recognized by ``RunLoader``."""
    sub = os.path.join(base, dir_name)
    _write_json(os.path.join(sub, 'benchmark_summary.json'), {'Request rate': request_rate})
    _write_json(os.path.join(sub, 'benchmark_percentile.json'), [])
    _write_json(os.path.join(sub, 'benchmark_args.json'), {'api': 'openai'})


class TestPerfReportLoader(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _make_run(self.tmp, 'parallel_2_number_5', request_rate=-1.0)
        _make_run(self.tmp, 'rate_1.5_number_4', request_rate=1.5)
        _make_run(self.tmp, 'rate_-1.0_number_30', request_rate=-1.0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _by_dir(self):
        from evalscope.perf.utils.report.perf_data import RunLoader
        runs = RunLoader.load_all(self.tmp, with_requests=False)
        return {r.dir_name: r for r in runs}

    def test_open_loop_no_fixed_rate_is_discovered(self):
        """rate_-1.0_number_30 must be loaded (previously silently dropped)."""
        runs = self._by_dir()
        self.assertIn('rate_-1.0_number_30', runs)
        run = runs['rate_-1.0_number_30']
        self.assertTrue(run.is_open_loop)
        self.assertEqual(run.name, 'Open-loop / Number 30')

    def test_fixed_rate_run_labeled_by_rate(self):
        run = self._by_dir()['rate_1.5_number_4']
        self.assertTrue(run.is_open_loop)
        self.assertEqual(run.name, 'Rate 1.5 rps / Number 4')

    def test_closed_loop_run_unaffected(self):
        run = self._by_dir()['parallel_2_number_5']
        self.assertFalse(run.is_open_loop)
        self.assertEqual(run.name, 'Parallel 2 / Number 5')

    def test_x_axis_detects_open_loop(self):
        from evalscope.perf.utils.report import perf_charts
        run = self._by_dir()['rate_-1.0_number_30']
        _, x_title = perf_charts._x_axis([run])
        self.assertEqual(x_title, 'Rate (req/s)')

    def test_x_axis_closed_loop(self):
        from evalscope.perf.utils.report import perf_charts
        run = self._by_dir()['parallel_2_number_5']
        _, x_title = perf_charts._x_axis([run])
        self.assertEqual(x_title, 'Concurrency')


if __name__ == '__main__':
    unittest.main()
