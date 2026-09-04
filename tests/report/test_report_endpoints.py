# Copyright (c) Alibaba, Inc. and its affiliates.
"""Route-level tests for the RESTful report resource endpoints.

Skipped automatically when Flask (service extra) is not installed.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

flask = pytest.importorskip('flask')  # noqa: F841  (service extra not installed → skip)

from evalscope.report import ReportRef  # noqa: E402
from evalscope.service.report_meta_cache import clear_report_meta_cache  # noqa: E402


class TestReportEndpoints(unittest.TestCase):

    def setUp(self):
        from evalscope.service.app import create_app

        self.tmp = tempfile.mkdtemp()
        self.client = create_app().test_client()
        clear_report_meta_cache()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_report_returns_bundle(self):
        report = mock.Mock()
        report.to_dict.return_value = {
            'schema_version': 2,
            'name': 'gsm8k',
            'dataset_name': 'gsm8k',
            'model_name': 'm',
            'metrics': [],
            'analysis': 'N/A',
            'judge_summary': None,
        }
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
        self.assertIsNone(res.get_json()['report_list'][0]['judge_summary'])

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
        frame = pd.DataFrame([{
            'Index': '0',
            'Input': 'question',
            'Metadata': {},
            'Generated': 'answer',
            'Gold': 'answer',
            'Pred': 'answer',
            'Score': {},
            'NScore': 1.0,
        }])
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
            refs = [
                ReportRef(run_id='20260101_120000', model_id='m1'),
                ReportRef(run_id='20260102_120000', model_id='m2'),
            ]
            with self.subTest(sort_by=sort_by, sort_order=sort_order), \
                    mock.patch('evalscope.service.blueprints.reports.scan_report_refs', return_value=refs), \
                    mock.patch('evalscope.service.blueprints.reports._build_report_meta', side_effect=case_items):
                clear_report_meta_cache()
                res = self.client.get(
                    '/api/v1/reports',
                    query_string={
                        'root_path': self.tmp,
                        'sort_by': sort_by,
                        'sort_order': sort_order
                    },
                )

            self.assertEqual([item['model_id'] for item in res.get_json()['reports']], expected)

    def test_report_list_rejects_removed_score_controls(self):
        cases = ({'sort_by': 'score'}, {'score_min': 0.5}, {'score_max': 0.9})

        for query in cases:
            with self.subTest(query=query):
                res = self.client.get('/api/v1/reports', query_string={'root_path': self.tmp, **query})

            self.assertEqual(res.status_code, 400)

    def test_report_list_rejects_unsupported_group_by(self):
        res = self.client.get('/api/v1/reports', query_string={'root_path': self.tmp, 'group_by': 'dataset'})

        self.assertEqual(res.status_code, 400)

    def test_report_list_group_by_model_rolls_up_same_model_reports_without_merging(self):
        items = [
            self._report_meta('gemma', run_id='run-a', dataset='mmmlu', timestamp='2026-08-17T22:09:00'),
            self._report_meta('gemma', run_id='run-b', dataset='hellaswag_hi', timestamp='2026-08-18T00:53:00'),
            self._report_meta('other-model', run_id='run-c', dataset='gsm8k', timestamp='2026-08-17T10:00:00'),
        ]
        refs = [ReportRef(run_id=item['run_id'], model_id=item['model_id']) for item in items]
        with mock.patch('evalscope.service.blueprints.reports.scan_report_refs', return_value=refs), \
                mock.patch('evalscope.service.blueprints.reports._build_report_meta', side_effect=items):
            clear_report_meta_cache()
            body = self.client.get(
                '/api/v1/reports',
                query_string={
                    'root_path': self.tmp, 'group_by': 'model', 'sort_by': 'model', 'sort_order': 'asc'
                },
            ).get_json()

        self.assertEqual(body['total'], 2)
        gemma_group, other_group = body['reports']

        self.assertEqual(gemma_group['model_name'], 'gemma')
        self.assertEqual(gemma_group['report_count'], 2)
        self.assertEqual(gemma_group['dataset_count'], 2)
        self.assertEqual(gemma_group['dataset_name'], 'hellaswag_hi, mmmlu')
        self.assertEqual(gemma_group['num_samples'], 2)
        # Most recently-run child sorts first; nothing about either child is altered.
        self.assertEqual(gemma_group['timestamp'], '2026-08-18T00:53:00')
        self.assertEqual(set(gemma_group['refs']), {'run-a/gemma', 'run-b/gemma'})
        self.assertEqual([c['run_id'] for c in gemma_group['children']], ['run-b', 'run-a'])
        self.assertEqual([c['dataset_name'] for c in gemma_group['children']], ['hellaswag_hi', 'mmmlu'])
        for child in gemma_group['children']:
            self.assertNotIn('_datasets', child)

        self.assertEqual(other_group['model_name'], 'other-model')
        self.assertEqual(other_group['report_count'], 1)

    def test_report_list_ungrouped_by_default(self):
        item = self._report_meta('model')
        refs = [ReportRef(run_id=item['run_id'], model_id=item['model_id'])]
        with mock.patch('evalscope.service.blueprints.reports.scan_report_refs', return_value=refs), \
                mock.patch('evalscope.service.blueprints.reports._build_report_meta', return_value=item):
            clear_report_meta_cache()
            body = self.client.get('/api/v1/reports', query_string={'root_path': self.tmp}).get_json()

        self.assertNotIn('report_count', body['reports'][0])

    def test_report_list_response_omits_score_comparability_fields(self):
        item = self._report_meta('model')
        ref = ReportRef(run_id='20260101_120000', model_id='model')
        with mock.patch('evalscope.service.blueprints.reports.scan_report_refs', return_value=[ref]), \
                mock.patch('evalscope.service.blueprints.reports._build_report_meta', return_value=item):
            body = self.client.get('/api/v1/reports', query_string={'root_path': self.tmp}).get_json()

        self.assertNotIn('score_comparable', body['filters'])
        self.assertNotIn('quality_ratio', body['reports'][0])

    def _seed_report_dir(self, run='20260101_120000', model='m', filename='gsm8k.json'):
        model_dir = Path(self.tmp) / run / 'reports' / model
        model_dir.mkdir(parents=True, exist_ok=True)
        report_file = model_dir / filename
        report_file.write_text('{}', encoding='utf-8')
        return report_file

    @staticmethod
    def _fake_report():
        return mock.Mock(
            dataset_name='gsm8k',
            dataset_pretty_name='GSM8K',
            num=5,
            model_name='m',
            primary_metric=None,
        )

    def test_report_list_includes_report_without_config_yaml(self):
        # Layer 0: the list reads only report JSONs, so a run with no
        # configs/*.yaml (which load_report_bundle rejects) still appears.
        self._seed_report_dir()
        with mock.patch(
            'evalscope.service.blueprints.reports.get_report_list',
            return_value=[self._fake_report()],
        ):
            res = self.client.get('/api/v1/reports', query_string={'root_path': self.tmp})

        self.assertEqual(res.status_code, 200)
        self.assertEqual([it['model_name'] for it in res.get_json()['reports']], ['m'])

    def test_report_meta_cache_reuses_unchanged_reports(self):
        # Layer A: a repeated request over unchanged files recomputes nothing;
        # editing a file busts that reference's entry.
        report_file = self._seed_report_dir()
        with mock.patch(
            'evalscope.service.blueprints.reports.get_report_list',
            return_value=[self._fake_report()],
        ) as get_list:
            self.client.get('/api/v1/reports', query_string={'root_path': self.tmp})
            self.client.get('/api/v1/reports', query_string={'root_path': self.tmp})
            reused = get_list.call_count
            report_file.write_text('{"changed": true}', encoding='utf-8')
            self.client.get('/api/v1/reports', query_string={'root_path': self.tmp})

        self.assertEqual(reused, 1)
        self.assertEqual(get_list.call_count, 2)

    def test_report_meta_cache_drops_deleted_reports(self):
        # Layer A: a reference that leaves the scan is pruned, so its return is a
        # fresh computation rather than a stale cache hit.
        self._seed_report_dir()
        ref = ReportRef(run_id='20260101_120000', model_id='m')
        with mock.patch(
            'evalscope.service.blueprints.reports.get_report_list',
            return_value=[self._fake_report()],
        ) as get_list:
            with mock.patch('evalscope.service.blueprints.reports.scan_report_refs', return_value=[ref]):
                self.client.get('/api/v1/reports', query_string={'root_path': self.tmp})
            with mock.patch('evalscope.service.blueprints.reports.scan_report_refs', return_value=[]):
                self.client.get('/api/v1/reports', query_string={'root_path': self.tmp})
            with mock.patch('evalscope.service.blueprints.reports.scan_report_refs', return_value=[ref]):
                self.client.get('/api/v1/reports', query_string={'root_path': self.tmp})

        self.assertEqual(get_list.call_count, 2)

    def test_report_list_supports_conditional_get(self):
        # Layer B: unchanged data revalidates as 304; changed data returns 200.
        report_file = self._seed_report_dir()
        with mock.patch(
            'evalscope.service.blueprints.reports.get_report_list',
            return_value=[self._fake_report()],
        ):
            first = self.client.get('/api/v1/reports', query_string={'root_path': self.tmp})
            etag = first.headers.get('ETag')
            self.assertTrue(etag)
            cached = self.client.get(
                '/api/v1/reports',
                query_string={'root_path': self.tmp},
                headers={'If-None-Match': etag},
            )
            self.assertEqual(cached.status_code, 304)
            report_file.write_text('{"x": 1}', encoding='utf-8')
            changed = self.client.get(
                '/api/v1/reports',
                query_string={'root_path': self.tmp},
                headers={'If-None-Match': etag},
            )

        self.assertEqual(changed.status_code, 200)

    @staticmethod
    def _report_meta(model_id, dataset='dataset', timestamp='2026-01-01T00:00:00', run_id='run'):
        return {
            'run_id': run_id,
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
