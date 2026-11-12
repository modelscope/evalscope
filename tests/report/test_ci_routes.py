"""HTTP-route regressions for the CI metadata (fourth-round review request).

A generated CI report (and an ordinary report) must survive the real
load/table service routes — not just the report model.
"""
import json
import tempfile
import unittest
from pathlib import Path

import pytest

flask = pytest.importorskip('flask')  # noqa: F841  (service extra not installed → skip)

from evalscope.api.metric import MetricSelector, SampleScore, Score  # noqa: E402
from evalscope.metrics.aggregators import Mean, MeanAndCI  # noqa: E402
from evalscope.report import CIInfo, Report  # noqa: E402
from evalscope.report.generator import ReportGenerator  # noqa: E402
from evalscope.report.report import ReportKey  # noqa: E402
from evalscope.service.report_meta_cache import clear_report_meta_cache  # noqa: E402


def _adapter():
    from types import SimpleNamespace

    return SimpleNamespace(
        name='gsm8k',
        pretty_name='GSM8K',
        description='',
        category_map={},
        primary_metric=MetricSelector(name='accuracy'),
    )


def _ci_report():
    scores = [
        SampleScore(sample_id=f's-{i}', score=Score(value={'accuracy': 1.0 if i % 4 else 0.0}))
        for i in range(20)
    ]
    return ReportGenerator.generate_report({'gsm8k': MeanAndCI()(scores)}, 'model', _adapter())


def _ordinary_report():
    scores = [SampleScore(sample_id=f's-{i}', score=Score(value={'accuracy': 1.0})) for i in range(40)]
    return ReportGenerator.generate_report({'gsm8k': Mean()(scores)}, 'model', _adapter())


class TestCIRoutes(unittest.TestCase):

    def setUp(self):
        from evalscope.service.app import create_app

        self.tmp = tempfile.mkdtemp()
        self.client = create_app().test_client()
        clear_report_meta_cache()
        # two separate runs: load_report_bundle globs every *.json under the model
        # directory, so mixing runs would put both reports into one report_list
        reports_dir = Path(self.tmp) / '20260101_120000' / 'reports' / 'model'
        reports_dir.mkdir(parents=True)
        configs_dir = Path(self.tmp) / '20260101_120000' / 'configs'
        configs_dir.mkdir(parents=True)
        (configs_dir / 'task.yaml').write_text('model: model\n', encoding='utf-8')
        self.ci_file = reports_dir / 'gsm8k.json'
        self.ci_file.write_text(json.dumps(_ci_report().to_dict()), encoding='utf-8')
        self.ordinary_run = '20260101_120001'
        ordinary_reports = Path(self.tmp) / self.ordinary_run / 'reports' / 'model'
        ordinary_reports.mkdir(parents=True)
        ordinary_configs = Path(self.tmp) / self.ordinary_run / 'configs'
        ordinary_configs.mkdir(parents=True)
        (ordinary_configs / 'task.yaml').write_text('model: model\n', encoding='utf-8')
        self.ordinary_file = ordinary_reports / 'gsm8k.json'
        self.ordinary_file.write_text(json.dumps(_ordinary_report().to_dict()), encoding='utf-8')

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _load(self, model='model'):
        return self.client.get(
            f'/api/v1/reports/runs/20260101_120000/models/{model}',
            query_string={'root_path': self.tmp},
        )

    def test_ci_report_loads_through_http_with_interval(self):
        res = self._load()
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        subset = payload['report_list'][0]['metrics'][0]['categories'][0]['subsets'][0]
        self.assertIsNotNone(subset['ci'])
        self.assertIn('low', subset['ci'])
        self.assertIn('high', subset['ci'])

    def test_ordinary_report_loads_without_ci_keys(self):
        res = self.client.get(
            f'/api/v1/reports/runs/{self.ordinary_run}/models/model',
            query_string={'root_path': self.tmp},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        for report in payload['report_list']:
            for metric in report['metrics']:
                for category in metric['categories']:
                    for subset in category['subsets']:
                        self.assertNotIn('ci', subset)
                        for key in subset:
                            self.assertFalse(key.startswith('ci_'), f'unexpected CI key {key}')

    def test_table_route_shows_interval_and_warning(self):
        # the actual CLI table renderer must show interval + warning together
        from evalscope.report import gen_table

        report = Report.model_validate_json(self.ci_file.read_text(encoding='utf-8'))
        table = gen_table(report_list=[report])
        lines = [line for line in table.splitlines() if 'gsm8k' in line.lower() or 'Accuracy' in line]
        interval_line = next((line for line in lines if '[' in line and 'low confidence' in line), None)
        self.assertIsNotNone(interval_line, f'no interval+warning line in table:\n{table}')

    def test_ordinary_dataframe_has_no_ci_column(self):
        report = Report.model_validate_json(self.ordinary_file.read_text(encoding='utf-8'))
        df = report.to_dataframe()
        self.assertNotIn(ReportKey.ci, df.columns)
