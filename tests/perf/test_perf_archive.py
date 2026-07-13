# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the historical perf-run archive service endpoints.

These tests build fake perf output directories in both the CLI layout
(``<ts>/<model>/parallel_*``) and the service layout (``<task_id>/perf/parallel_*``)
and exercise the ``/api/v1/perf/{list,detail,chart,history/report}`` endpoints.

Skipped automatically when Flask (service extra) is not installed.
"""
import json
import os
import pytest
import tempfile
import unittest

flask = pytest.importorskip('flask')  # noqa: F841  (service extra not installed → skip)


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def _make_run(run_dir: str, *, with_html: bool) -> None:
    """Create a minimal perf-run directory with one parallel_* sub-run."""
    sub = os.path.join(run_dir, 'parallel_1_number_2')
    _write_json(os.path.join(sub, 'benchmark_summary.json'), {})
    _write_json(os.path.join(sub, 'benchmark_percentile.json'), [])
    _write_json(
        os.path.join(sub, 'benchmark_args.json'), {
            'model': 'my-model',
            'api': 'openai',
            'dataset': 'openqa',
        }
    )
    if with_html:
        with open(os.path.join(run_dir, 'perf_report.html'), 'w', encoding='utf-8') as f:
            f.write('<html><body>perf report</body></html>')


class TestPerfArchive(unittest.TestCase):

    def setUp(self):
        from evalscope.service.app import create_app

        self.tmp = tempfile.mkdtemp()
        # CLI layout: <ts>/<model>/  (with generated HTML report)
        self.cli_rel = os.path.join('20260101_120000', 'my-model')
        _make_run(os.path.join(self.tmp, self.cli_rel), with_html=True)
        # Service layout: <task_id>/perf/  (no pre-generated HTML)
        self.svc_rel = os.path.join('task_abc', 'perf')
        _make_run(os.path.join(self.tmp, self.svc_rel), with_html=False)

        self.client = create_app().test_client()

    def test_list_finds_both_layouts(self):
        res = self.client.get('/api/v1/perf/list', query_string={'root_path': self.tmp})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        paths = {r['path'] for r in body['runs']}
        self.assertIn(self.cli_rel, paths)
        self.assertIn(self.svc_rel, paths)
        self.assertEqual(body['total'], 2)

    def test_detail_returns_summary_fields(self):
        res = self.client.get('/api/v1/perf/detail', query_string={'root_path': self.tmp, 'path': self.cli_rel})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body['model'], 'my-model')
        self.assertIn('summary_columns', body)
        self.assertIn('summary_rows', body)
        self.assertEqual(body['num_runs'], 1)

    def test_history_report_serves_existing_html(self):
        res = self.client.get('/api/v1/perf/history/report', query_string={'root_path': self.tmp, 'path': self.cli_rel})
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'perf report', res.data)

    def test_path_traversal_is_rejected(self):
        res = self.client.get('/api/v1/perf/detail', query_string={'root_path': self.tmp, 'path': '../../etc'})
        self.assertEqual(res.status_code, 400)

    def test_chart_returns_plotly_html(self):
        pytest.importorskip('plotly')
        res = self.client.get(
            '/api/v1/perf/chart',
            query_string={
                'root_path': self.tmp,
                'path': self.cli_rel,
                'chart_type': 'latency'
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn('text/html', res.headers.get('Content-Type', ''))


if __name__ == '__main__':
    unittest.main()
