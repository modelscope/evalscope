# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the report rename endpoint
(POST /api/v1/reports/runs/{run_id}/models/{model_id}/rename).

Renames a report's model in place and verifies the directory move, the
rewritten ``model_name`` field inside each dataset result file, and the
usual validation/guard rejections.

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


def _report_dict(model_name: str, dataset_name: str, num: int = 10, score: float = 0.5) -> dict:
    return {
        'name': f'{model_name}@{dataset_name}',
        'dataset_name': dataset_name,
        'model_name': model_name,
        'metrics': [{
            'name': 'acc',
            'categories': [{
                'name': ['default'],
                'subsets': [{
                    'name': 'default',
                    'score': score,
                    'num': num,
                }],
            }],
        }],
    }


def _write_json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


class TestReportRename(unittest.TestCase):

    def setUp(self):
        from evalscope.service.app import create_app

        self.tmp = tempfile.mkdtemp()
        self.prefix = '20260101_120000'
        self.run_dir = os.path.join(self.tmp, self.prefix)
        _write_json(os.path.join(self.run_dir, 'reports', 'model-a', 'gsm8k.json'), _report_dict('model-a', 'gsm8k'))
        _write_json(os.path.join(self.run_dir, 'predictions', 'model-a', 'gsm8k_default.jsonl'), {})
        _write_json(os.path.join(self.run_dir, 'reviews', 'model-a', 'gsm8k_default.jsonl'), {})
        os.makedirs(os.path.join(self.run_dir, 'configs'), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'logs'), exist_ok=True)
        self.client = create_app().test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rename(self, run_id, model_id, new_model_name, root=None):
        return self.client.post(
            f'/api/v1/reports/runs/{run_id}/models/{model_id}/rename',
            json={
                'root_path': root if root is not None else self.tmp,
                'new_model_name': new_model_name,
            },
        )

    def test_rename_success_moves_dirs_and_rewrites_model_name(self):
        res = self._rename(self.prefix, 'model-a', 'model-a-v2')
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body['success'])
        self.assertEqual(body['run_id'], self.prefix)
        self.assertEqual(body['model_id'], 'model-a-v2')

        self.assertFalse(os.path.isdir(os.path.join(self.run_dir, 'reports', 'model-a')))
        new_report_json = os.path.join(self.run_dir, 'reports', 'model-a-v2', 'gsm8k.json')
        self.assertTrue(os.path.isfile(new_report_json))
        with open(new_report_json, 'r', encoding='utf-8') as f:
            self.assertEqual(json.load(f)['model_name'], 'model-a-v2')

        self.assertTrue(os.path.isdir(os.path.join(self.run_dir, 'predictions', 'model-a-v2')))
        self.assertTrue(os.path.isdir(os.path.join(self.run_dir, 'reviews', 'model-a-v2')))

    def test_rename_rejects_missing_new_name(self):
        res = self._rename(self.prefix, 'model-a', '')
        self.assertEqual(res.status_code, 400)

    def test_rename_rejects_path_separator_in_new_name(self):
        res = self._rename(self.prefix, 'model-a', '../escape')
        self.assertEqual(res.status_code, 400)

    def test_rename_rejects_same_name(self):
        res = self._rename(self.prefix, 'model-a', 'model-a')
        self.assertEqual(res.status_code, 400)

    def test_rename_rejects_collision_with_existing_model(self):
        _write_json(os.path.join(self.run_dir, 'reports', 'model-b', 'gsm8k.json'), _report_dict('model-b', 'gsm8k'))
        res = self._rename(self.prefix, 'model-a', 'model-b')
        self.assertEqual(res.status_code, 409)
        # The source model report must be untouched by the rejected rename.
        self.assertTrue(os.path.isdir(os.path.join(self.run_dir, 'reports', 'model-a')))

    def test_rename_rejects_running_task(self):
        with mock.patch('evalscope.service.blueprints.reports.active_task_ids', return_value={self.prefix}):
            res = self._rename(self.prefix, 'model-a', 'model-a-v2')
        self.assertEqual(res.status_code, 409)

    def test_rename_rejects_unknown_report(self):
        res = self._rename(self.prefix, 'ghost', 'model-a-v2')
        self.assertEqual(res.status_code, 404)

    def test_rename_rejects_malformed_reference(self):
        res = self._rename('..', 'model-a', 'model-a-v2')
        self.assertEqual(res.status_code, 400)


if __name__ == '__main__':
    unittest.main()
