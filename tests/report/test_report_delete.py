# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the eval-report delete endpoint (DELETE /api/v1/reports/report).

Builds a fake eval output tree with two model reports sharing one run
directory and verifies per-model deletion, run-dir cleanup, path-traversal
rejection and running-task protection.

Skipped automatically when Flask (service extra) is not installed.
"""
import json
import os
import pytest
import shutil
import tempfile
import unittest
from unittest import mock

flask = pytest.importorskip('flask')  # noqa: F841  (service extra not installed → skip)


def _write_json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


class TestReportDelete(unittest.TestCase):

    def setUp(self):
        from evalscope.service.app import create_app

        self.tmp = tempfile.mkdtemp()
        self.prefix = '20260101_120000'
        run_dir = os.path.join(self.tmp, self.prefix)
        # Two models share the run directory; each has reports + predictions.
        for model in ('model-a', 'model-b'):
            _write_json(os.path.join(run_dir, 'reports', model, 'gsm8k.json'), {})
            _write_json(os.path.join(run_dir, 'predictions', model, 'gsm8k.jsonl'), {})
        with open(os.path.join(run_dir, 'reports', 'report.html'), 'w', encoding='utf-8') as f:
            f.write('<html><body>model-a model-b</body></html>')
        os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'configs'), exist_ok=True)

        self.client = create_app().test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _delete(self, report_name: str):
        return self.client.delete(
            '/api/v1/reports/report',
            query_string={
                'root_path': self.tmp,
                'report_name': report_name
            },
        )

    def test_delete_one_model_keeps_run_dir(self):
        res = self._delete(f'{self.prefix}@@model-a::gsm8k')
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body['success'])
        run_dir = os.path.join(self.tmp, self.prefix)
        # model-a artefacts are gone; model-b and the run dir survive.
        self.assertFalse(os.path.isdir(os.path.join(run_dir, 'reports', 'model-a')))
        self.assertFalse(os.path.isdir(os.path.join(run_dir, 'predictions', 'model-a')))
        self.assertTrue(os.path.isdir(os.path.join(run_dir, 'reports', 'model-b')))
        self.assertTrue(os.path.isdir(run_dir))
        html_report = os.path.join(run_dir, 'reports', 'report.html')
        if os.path.isfile(html_report):
            with open(html_report, 'r', encoding='utf-8') as f:
                self.assertNotIn('model-a', f.read())

    def test_delete_last_model_removes_run_dir(self):
        self._delete(f'{self.prefix}@@model-a::gsm8k')
        res = self._delete(f'{self.prefix}@@model-b::gsm8k')
        self.assertEqual(res.status_code, 200)
        self.assertFalse(os.path.isdir(os.path.join(self.tmp, self.prefix)))
        # The outputs root itself is never removed.
        self.assertTrue(os.path.isdir(self.tmp))

    def test_delete_rejects_bad_names(self):
        # Missing report_name.
        res = self.client.delete('/api/v1/reports/report', query_string={'root_path': self.tmp})
        self.assertEqual(res.status_code, 400)
        # Malformed identifier (no @@ / :: tokens).
        self.assertEqual(self._delete('not-a-report-name').status_code, 400)
        # Path traversal in the prefix segment.
        self.assertEqual(self._delete('../../etc@@model-a::gsm8k').status_code, 400)
        # Unknown model report.
        self.assertEqual(self._delete(f'{self.prefix}@@ghost::gsm8k').status_code, 404)
        # The tree is untouched by all rejected attempts.
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, self.prefix, 'reports', 'model-a')))

    def test_delete_rejects_running_task(self):
        with mock.patch('evalscope.service.blueprints.reports.active_task_ids', return_value={self.prefix}):
            res = self._delete(f'{self.prefix}@@model-a::gsm8k')
        self.assertEqual(res.status_code, 409)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, self.prefix, 'reports', 'model-a')))


if __name__ == '__main__':
    unittest.main()
