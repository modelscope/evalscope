# Copyright (c) Alibaba, Inc. and its affiliates.
"""Route-level tests for the RESTful report resource endpoints.

Skipped automatically when Flask (service extra) is not installed.
"""
import pytest
import tempfile
import unittest
from pathlib import Path
from unittest import mock

flask = pytest.importorskip('flask')  # noqa: F841  (service extra not installed → skip)


class TestReportEndpoints(unittest.TestCase):

    def setUp(self):
        from evalscope.service.app import create_app

        self.tmp = tempfile.mkdtemp()
        self.client = create_app().test_client()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_report_returns_bundle(self):
        report = mock.Mock()
        report.to_dict.return_value = {'schema_version': 2, 'name': 'gsm8k'}
        with mock.patch(
            'evalscope.service.blueprints.reports.load_report_bundle',
            return_value=([report], ['gsm8k'], {
                'model': 'm'
            }),
        ):
            res = self.client.get(
                '/api/v1/reports/runs/20260101_120000/models/model-a',
                query_string={'root_path': self.tmp},
            )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()['datasets'], ['gsm8k'])

    def test_load_report_missing_run_returns_404(self):
        # No config yaml under the (absent) run directory: load_report_bundle raises FileNotFoundError.
        res = self.client.get(
            '/api/v1/reports/runs/20990101_000000/models/ghost',
            query_string={'root_path': self.tmp},
        )
        self.assertEqual(res.status_code, 404)

    def test_load_report_missing_model_returns_404(self):
        configs_dir = Path(self.tmp) / '20260101_120000' / 'configs'
        configs_dir.mkdir(parents=True)
        (configs_dir / 'task.yaml').write_text('model: m\n', encoding='utf-8')

        res = self.client.get(
            '/api/v1/reports/runs/20260101_120000/models/ghost',
            query_string={'root_path': self.tmp},
        )
        self.assertEqual(res.status_code, 404)

    def test_predictions_requires_dataset_and_subset(self):
        res = self.client.get(
            '/api/v1/reports/runs/20260101_120000/models/model-a/predictions',
            query_string={
                'root_path': self.tmp,
                'dataset_name': 'gsm8k'
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_predictions_returns_rows(self):
        import pandas as pd
        frame = pd.DataFrame([{'Index': '0', 'NScore': 1.0}])
        with mock.patch(
            'evalscope.service.blueprints.reports.get_model_prediction',
            return_value=frame,
        ):
            res = self.client.get(
                '/api/v1/reports/runs/20260101_120000/models/model-a/predictions',
                query_string={
                    'root_path': self.tmp,
                    'dataset_name': 'gsm8k',
                    'subset_name': 'main'
                },
            )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.get_json()['predictions']), 1)

    def test_report_list_sorts_by_supported_fields(self):
        items = [
            self._report_meta('z-model', dataset='a-dataset', timestamp='2026-01-01T00:00:00'),
            self._report_meta('a-model', dataset='z-dataset', timestamp='2026-01-02T00:00:00'),
        ]
        cases = [
            ('model', 'asc', ['a-model', 'z-model']),
            ('dataset', 'asc', ['z-model', 'a-model']),
            ('time', 'desc', ['a-model', 'z-model']),
        ]

        for sort_by, sort_order, expected in cases:
            case_items = [{**item, '_datasets': list(item['_datasets'])} for item in items]
            with self.subTest(sort_by=sort_by, sort_order=sort_order), \
                    mock.patch('evalscope.service.blueprints.reports.scan_report_refs', return_value=range(2)), \
                    mock.patch('evalscope.service.blueprints.reports._build_report_meta', side_effect=case_items):
                res = self.client.get(
                    '/api/v1/reports',
                    query_string={
                        'root_path': self.tmp, 'sort_by': sort_by, 'sort_order': sort_order
                    },
                )

            self.assertEqual([item['model_id'] for item in res.get_json()['reports']], expected)

    def test_report_list_rejects_removed_score_controls(self):
        cases = ({'sort_by': 'score'}, {'score_min': 0.5}, {'score_max': 0.9})

        for query in cases:
            with self.subTest(query=query):
                res = self.client.get('/api/v1/reports', query_string={'root_path': self.tmp, **query})

            self.assertEqual(res.status_code, 400)

    def test_report_list_response_omits_score_comparability_fields(self):
        item = self._report_meta('model')
        with mock.patch('evalscope.service.blueprints.reports.scan_report_refs', return_value=range(1)), \
                mock.patch('evalscope.service.blueprints.reports._build_report_meta', return_value=item):
            body = self.client.get('/api/v1/reports', query_string={'root_path': self.tmp}).get_json()

        self.assertNotIn('score_comparable', body['filters'])
        self.assertNotIn('quality_ratio', body['reports'][0])

    @staticmethod
    def _report_meta(model_id, dataset='dataset', timestamp='2026-01-01T00:00:00'):
        return {
            'run_id': 'run',
            'model_id': model_id,
            'model_name': model_id,
            'dataset_name': dataset,
            'dataset_pretty_name': dataset.title(),
            'num_samples': 1,
            'timestamp': timestamp,
            'primary_metrics': [],
            '_datasets': [dataset],
        }


if __name__ == '__main__':
    unittest.main()
