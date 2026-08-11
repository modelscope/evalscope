# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the report merge endpoint (POST /api/v1/reports/merge).

Builds two (or more) fake eval output trees for the same model with disjoint
datasets and verifies successful merge, same-model validation, dataset-overlap
rejection, running-task protection and malformed/unknown references.

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


def _write_run(run_dir: str, model: str, datasets: list, task_config: dict = None) -> None:
    for dataset in datasets:
        _write_json(os.path.join(run_dir, 'reports', model, f'{dataset}.json'), _report_dict(model, dataset))
        _write_json(os.path.join(run_dir, 'predictions', model, f'{dataset}_default.jsonl'), {})
        _write_json(os.path.join(run_dir, 'reviews', model, f'{dataset}_default.jsonl'), {})
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    configs_dir = os.path.join(run_dir, 'configs')
    os.makedirs(configs_dir, exist_ok=True)
    cfg = task_config if task_config is not None else {'model': model, 'datasets': datasets}
    import yaml
    with open(os.path.join(configs_dir, 'task_config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f)


class TestReportMerge(unittest.TestCase):

    def setUp(self):
        from evalscope.service.app import create_app

        self.tmp = tempfile.mkdtemp()
        self.prefix_a = '20260101_120000'
        self.prefix_b = '20260102_120000'
        _write_run(os.path.join(self.tmp, self.prefix_a), 'model-a', ['gsm8k'])
        _write_run(os.path.join(self.tmp, self.prefix_b), 'model-a', ['mmlu'])
        self.client = create_app().test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _merge(self, refs, root=None):
        return self.client.post(
            '/api/v1/reports/merge',
            json={
                'root_path': root if root is not None else self.tmp,
                'refs': refs,
            },
        )

    def test_merge_success_combines_disjoint_datasets(self):
        res = self._merge([
            f'{self.prefix_a}/model-a',
            f'{self.prefix_b}/model-a',
        ])
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body['success'])
        new_run_id = body['run_id']
        self.assertEqual(body['model_id'], 'model-a')

        new_run_dir = os.path.join(self.tmp, new_run_id)
        self.assertTrue(os.path.isfile(os.path.join(new_run_dir, 'reports', 'model-a', 'gsm8k.json')))
        self.assertTrue(os.path.isfile(os.path.join(new_run_dir, 'reports', 'model-a', 'mmlu.json')))
        self.assertTrue(os.path.isfile(os.path.join(new_run_dir, 'predictions', 'model-a', 'gsm8k_default.jsonl')))
        self.assertTrue(os.path.isfile(os.path.join(new_run_dir, 'reviews', 'model-a', 'mmlu_default.jsonl')))
        self.assertTrue(os.path.isfile(os.path.join(new_run_dir, 'configs', 'task_config.yaml')))
        self.assertTrue(os.path.isfile(os.path.join(new_run_dir, 'progress.json')))

        # Original run directories are untouched (merge copies, never moves).
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, self.prefix_a, 'reports', 'model-a')))
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, self.prefix_b, 'reports', 'model-a')))

        # The merged report is independently loadable through the normal API.
        load_res = self.client.get(
            f'/api/v1/reports/runs/{new_run_id}/models/model-a', query_string={'root_path': self.tmp}
        )
        self.assertEqual(load_res.status_code, 200)
        load_body = load_res.get_json()
        self.assertEqual(len(load_body['report_list']), 2)

    def test_merge_rejects_fewer_than_two_reports(self):
        res = self._merge([f'{self.prefix_a}/model-a'])
        self.assertEqual(res.status_code, 400)

    def test_merge_rejects_different_models(self):
        _write_run(os.path.join(self.tmp, '20260103_120000'), 'model-b', ['arc'])
        res = self._merge([
            f'{self.prefix_a}/model-a',
            '20260103_120000/model-b',
        ])
        self.assertEqual(res.status_code, 400)

    def test_merge_rejects_overlapping_datasets(self):
        _write_run(os.path.join(self.tmp, '20260104_120000'), 'model-a', ['gsm8k'])
        res = self._merge([
            f'{self.prefix_a}/model-a',
            '20260104_120000/model-a',
        ])
        self.assertEqual(res.status_code, 409)
        # Nothing was left behind by the aborted merge.
        entries = os.listdir(self.tmp)
        self.assertEqual([e for e in entries if e.startswith('merged_')], [])

    def test_merge_rejects_running_task(self):
        with mock.patch('evalscope.service.blueprints.reports.active_task_ids', return_value={self.prefix_a}):
            res = self._merge([
                f'{self.prefix_a}/model-a',
                f'{self.prefix_b}/model-a',
            ])
        self.assertEqual(res.status_code, 409)

    def test_merge_rejects_unknown_report(self):
        # Same model_id (passes the model-equality check) but the second
        # reference's run directory doesn't exist on disk.
        res = self._merge([
            f'{self.prefix_a}/model-a',
            'no_such_run/model-a',
        ])
        self.assertEqual(res.status_code, 404)

    def test_merge_rejects_malformed_ref(self):
        res = self._merge(['not-a-report-ref', f'{self.prefix_a}/model-a'])
        self.assertEqual(res.status_code, 400)

    def test_merge_rejects_validation_failure_without_touching_disk(self):
        before = set(os.listdir(self.tmp))
        self._merge([f'{self.prefix_a}/model-a', f'{self.prefix_a}/ghost'])
        after = set(os.listdir(self.tmp))
        self.assertEqual(before, after)

    def test_merge_rolls_back_partial_directory_on_copy_failure(self):
        before = set(os.listdir(self.tmp))
        with mock.patch('evalscope.service.blueprints.reports.shutil.copy2', side_effect=OSError('disk full')):
            res = self._merge([
                f'{self.prefix_a}/model-a',
                f'{self.prefix_b}/model-a',
            ])
        self.assertEqual(res.status_code, 500)
        after = set(os.listdir(self.tmp))
        # The half-written merged_* run directory must be cleaned up, not left behind.
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
