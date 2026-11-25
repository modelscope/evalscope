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
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


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

    def test_compare_chart_uses_per_trace_markers_and_full_hover_identity(self):
        from unittest import mock

        from evalscope.perf.utils.report import perf_charts
        run = self._by_dir()['parallel_2_number_5']
        with mock.patch.object(perf_charts.ChartBuilder, 'line', return_value='chart') as line:
            perf_charts.build_compare_chart(
                [('R1 · model', [run])],
                'latency',
                series_details=['model · dataset · path'],
            )

        trace = line.call_args.args[0][0]
        self.assertEqual(trace['mode'], 'markers')
        self.assertFalse(trace['connectgaps'])
        self.assertEqual(trace['customdata'][0][0], 'model · dataset · path')
        self.assertIn('Requests=%{customdata[2]}', trace['hovertemplate'])

    def test_input_length_compare_chart_sorts_and_preserves_gaps(self):
        from unittest import mock

        from evalscope.perf.utils.report import perf_charts
        run = self._by_dir()['parallel_2_number_5']
        points = [
            (2048, 1, run, 'R2'),
            (1024, 1, run, 'R1'),
            (4096, 2, run, 'R3'),
        ]
        with mock.patch.object(perf_charts.ChartBuilder, 'line', return_value='chart') as line:
            perf_charts.build_input_length_compare_chart(points, 'latency')

        traces = line.call_args.args[0]
        self.assertEqual([trace['name'] for trace in traces], ['C=1', 'C=2'])
        self.assertEqual(traces[0]['x'], [1024, 2048, 4096])
        self.assertEqual(traces[0]['y'][2], None)
        self.assertEqual(traces[1]['y'][:2], [None, None])
        self.assertEqual(traces[0]['mode'], 'lines+markers')
        self.assertEqual(traces[1]['mode'], 'markers')
        self.assertTrue(all(trace['connectgaps'] is False for trace in traces))
        self.assertNotIn('dtick', line.call_args.kwargs['extra_layout']['xaxis'])

    def test_html_percentiles_distinguish_missing_values_from_zero(self) -> None:
        from evalscope.perf.utils.report.generate_report import gen_perf_html_report

        _write_json(
            os.path.join(self.tmp, 'parallel_2_number_5', 'benchmark_percentile.json'),
            [{'Percentiles': '99%', 'Latency (s)': 0, 'TTFT (ms)': float('nan'), 'TPOT (ms)': float('inf')}],
        )
        report = gen_perf_html_report(self.tmp, {}, SimpleNamespace(api='openai'))
        cells = re.findall(r'<td class="num-cell">([^<]*)</td>', Path(report).read_text())
        self.assertIn('0.0', cells)
        self.assertIn('—', cells)
        self.assertNotIn('nan', cells)
        self.assertNotIn('inf', cells)


if __name__ == '__main__':
    unittest.main()
