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
import shutil
import sqlite3
import tempfile
import unittest

flask = pytest.importorskip('flask')  # noqa: F841  (service extra not installed → skip)


def _write_json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def _make_run(run_dir: str, *, with_html: bool, with_tokens: bool = True) -> None:
    """Create a minimal perf-run directory with one parallel_* sub-run."""
    sub = os.path.join(run_dir, 'parallel_1_number_2')
    summary = {
        'Total Requests': 2,
        'Success Requests': 2,
    }
    if with_tokens:
        summary.update({
            'Avg Input Tokens': 10000.0,
            'Avg Output Tokens': 300.0,
        })
    _write_json(os.path.join(sub, 'benchmark_summary.json'), summary)
    _write_json(os.path.join(sub, 'benchmark_percentile.json'), [])
    _write_json(
        os.path.join(sub, 'benchmark_args.json'), {
            'model': 'my-model',
            'api': 'openai',
            'dataset': 'openqa',
            'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        }
    )
    if with_html:
        with open(os.path.join(run_dir, 'perf_report.html'), 'w', encoding='utf-8') as f:
            f.write('<html><body>perf report</body></html>')


def _make_db(sub_dir: str, *, n_success: int, n_failed: int) -> None:
    """Create a minimal ``benchmark_data.db`` with a ``result`` table."""
    os.makedirs(sub_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(sub_dir, 'benchmark_data.db'))
    try:
        conn.execute(
            'CREATE TABLE result ('
            'start_time REAL, completed_time REAL, latency REAL, '
            'first_chunk_latency REAL, prompt_tokens INTEGER, completion_tokens INTEGER, '
            'inter_token_latencies TEXT, time_per_output_token REAL, success INTEGER, is_stream INTEGER)'
        )
        rows = [(0.0, 1.0, 0.5, 0.1, 10, 20, '[]', 0.01, 1, 1) for _ in range(n_success)]
        rows += [(0.0, 1.0, 0.5, None, 10, 0, '[]', None, 0, 1) for _ in range(n_failed)]
        conn.executemany(
            'INSERT INTO result (start_time, completed_time, latency, first_chunk_latency, '
            'prompt_tokens, completion_tokens, inter_token_latencies, time_per_output_token, success, is_stream) '
            'VALUES (?,?,?,?,?,?,?,?,?,?)',
            rows,
        )
        conn.commit()
    finally:
        conn.close()


class TestPerfArchive(unittest.TestCase):

    def setUp(self):
        from evalscope.service.app import create_app

        self.tmp = tempfile.mkdtemp()
        # CLI layout: <ts>/<model>/  (with generated HTML report)
        self.cli_rel = os.path.join('20260101_120000', 'my-model')
        _make_run(os.path.join(self.tmp, self.cli_rel), with_html=True)
        # Give the CLI run a per-request DB so /runs + /requests can be exercised.
        self.n_success, self.n_failed = 5, 2
        self.n_total = self.n_success + self.n_failed
        _make_db(
            os.path.join(self.tmp, self.cli_rel, 'parallel_1_number_2'),
            n_success=self.n_success,
            n_failed=self.n_failed,
        )
        # Service layout: <task_id>/perf/  (no pre-generated HTML)
        self.svc_rel = os.path.join('task_abc', 'perf')
        _make_run(os.path.join(self.tmp, self.svc_rel), with_html=False)

        self.client = create_app().test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_finds_both_layouts(self):
        res = self.client.get('/api/v1/perf/list', query_string={'root_path': self.tmp})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        paths = {r['path'] for r in body['runs']}
        self.assertIn(self.cli_rel, paths)
        self.assertIn(self.svc_rel, paths)
        self.assertEqual(body['total'], 2)
        cli_run = next(run for run in body['runs'] if run['path'] == self.cli_rel)
        self.assertEqual(cli_run['api_host'], 'dashscope.aliyuncs.com')
        self.assertEqual(cli_run['concurrency'], [1])

    def test_detail_returns_summary_fields(self):
        res = self.client.get('/api/v1/perf/detail', query_string={'root_path': self.tmp, 'path': self.cli_rel})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body['model'], 'my-model')
        self.assertIn('summary_columns', body)
        self.assertIn('summary_rows', body)
        self.assertEqual(body['summary_columns'][0], {
            'key': 'concurrency', 'label': 'Conc.', 'semantics': None
        })
        latency = next(column for column in body['summary_columns'] if column['key'] == 'avg_latency')
        self.assertEqual(latency['label'], 'Avg Lat.(s)')
        self.assertEqual(latency['semantics']['semantic_id'], 'perf.latency.seconds')
        self.assertIsInstance(body['summary_rows'][0]['values']['avg_latency'], (int, float))
        self.assertEqual(body['summary_rows'][0]['sample_counts']['avg_latency'], self.n_success)
        self.assertEqual(body['summary_rows'][0]['sample_counts']['p99_ttft'], self.n_success)
        self.assertEqual(body['summary_rows'][0]['sample_counts']['success_rate'], self.n_total)
        self.assertEqual(body['total_requests'], 2)
        self.assertNotIn('summary_sample_counts', body)
        self.assertNotIn('metric_semantics', body)
        self.assertEqual(body['num_runs'], 1)
        self.assertEqual(body['basic_info']['API Host'], 'dashscope.aliyuncs.com')

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

    def test_runs_reports_request_count(self):
        res = self.client.get('/api/v1/perf/runs', query_string={'root_path': self.tmp, 'path': self.cli_rel})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body['total'], 1)
        run = body['runs'][0]
        self.assertEqual(run['dir_name'], 'parallel_1_number_2')
        self.assertTrue(run['has_requests'])
        self.assertEqual(run['num_requests'], self.n_total)

    def test_requests_pagination_and_status_filter(self):
        base = {'root_path': self.tmp, 'path': self.cli_rel, 'run': 'parallel_1_number_2'}
        # First page of size 3 out of 7 total.
        res = self.client.get('/api/v1/perf/requests', query_string={**base, 'page': 1, 'page_size': 3})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body['total'], self.n_total)
        self.assertEqual(len(body['rows']), 3)
        self.assertTrue(body['has_db'])
        self.assertEqual(body['rows'][0]['#'], 1)
        # Last page carries the remainder.
        res2 = self.client.get('/api/v1/perf/requests', query_string={**base, 'page': 3, 'page_size': 3})
        self.assertEqual(len(res2.get_json()['rows']), 1)
        # Status filter narrows the total (but has_db stays True).
        res3 = self.client.get('/api/v1/perf/requests', query_string={**base, 'status': 'failed'})
        self.assertEqual(res3.get_json()['total'], self.n_failed)
        self.assertTrue(res3.get_json()['has_db'])

    def test_requests_rejects_unknown_run(self):
        res = self.client.get(
            '/api/v1/perf/requests',
            query_string={
                'root_path': self.tmp,
                'path': self.cli_rel,
                'run': 'not_a_run'
            },
        )
        self.assertEqual(res.status_code, 404)

    def test_metadata_only_load_skips_requests(self):
        from evalscope.perf.utils.report.perf_data import RunLoader
        run_dir = os.path.join(self.tmp, self.cli_rel)
        runs = RunLoader.load_all(run_dir, with_requests=False)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].requests, [])
        sub = os.path.join(run_dir, 'parallel_1_number_2')
        self.assertEqual(RunLoader.count_requests(sub), self.n_total)
        recs, total = RunLoader.query_requests(sub, offset=0, limit=3)
        self.assertEqual(total, self.n_total)
        self.assertEqual(len(recs), 3)
        failed, failed_total = RunLoader.query_requests(sub, status='failed', offset=0, limit=100)
        self.assertEqual(failed_total, self.n_failed)
        self.assertTrue(all(not r.success for r in failed))

    def test_list_includes_token_fields(self):
        res = self.client.get('/api/v1/perf/list', query_string={'root_path': self.tmp})
        self.assertEqual(res.status_code, 200)
        cli_run = next(run for run in res.get_json()['runs'] if run['path'] == self.cli_rel)
        self.assertEqual(cli_run['avg_input_tokens'], 10000.0)
        self.assertEqual(cli_run['avg_output_tokens'], 300.0)

    def test_list_omits_missing_token_fields(self):
        legacy_rel = os.path.join('20260102_120000', 'legacy-model')
        _make_run(os.path.join(self.tmp, legacy_rel), with_html=False, with_tokens=False)

        res = self.client.get('/api/v1/perf/list', query_string={'root_path': self.tmp})
        self.assertEqual(res.status_code, 200)
        legacy_run = next(run for run in res.get_json()['runs'] if run['path'] == legacy_rel)
        self.assertNotIn('avg_input_tokens', legacy_run)
        self.assertNotIn('avg_output_tokens', legacy_run)

    def test_delete_run(self):
        res = self.client.delete('/api/v1/perf/run', query_string={'root_path': self.tmp, 'path': self.svc_rel})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body['success'])
        self.assertEqual(body['path'], self.svc_rel)
        self.assertFalse(os.path.isdir(os.path.join(self.tmp, self.svc_rel)))
        # The now-empty task_id parent directory is pruned as well.
        self.assertFalse(os.path.isdir(os.path.join(self.tmp, 'task_abc')))
        # The outputs root itself is never removed.
        self.assertTrue(os.path.isdir(self.tmp))
        # The deleted run no longer shows up in the list.
        res2 = self.client.get('/api/v1/perf/list', query_string={'root_path': self.tmp})
        paths = {run['path'] for run in res2.get_json()['runs']}
        self.assertNotIn(self.svc_rel, paths)
        self.assertIn(self.cli_rel, paths)

    def test_delete_rejects_traversal_and_non_run_dirs(self):
        # Path traversal outside the root is rejected.
        res = self.client.delete('/api/v1/perf/run', query_string={'root_path': self.tmp, 'path': '../../etc'})
        self.assertEqual(res.status_code, 400)
        # A directory that is not a perf run (the task_id parent) is rejected.
        res2 = self.client.delete('/api/v1/perf/run', query_string={'root_path': self.tmp, 'path': 'task_abc'})
        self.assertEqual(res2.status_code, 400)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, 'task_abc')))
        # Missing path is rejected.
        res3 = self.client.delete('/api/v1/perf/run', query_string={'root_path': self.tmp})
        self.assertEqual(res3.status_code, 400)

    def test_delete_rejects_running_task(self):
        from unittest import mock
        with mock.patch('evalscope.service.blueprints.perf.active_task_ids', return_value={'task_abc'}):
            res = self.client.delete('/api/v1/perf/run', query_string={'root_path': self.tmp, 'path': self.svc_rel})
        self.assertEqual(res.status_code, 409)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, self.svc_rel)))


if __name__ == '__main__':
    unittest.main()
