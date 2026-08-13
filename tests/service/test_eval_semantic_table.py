import pytest

pytest.importorskip('flask')

from evalscope.api.metric import AggScore
from evalscope.api.metric.semantics import MetricSelector
from evalscope.report.generator import ReportGenerator
from evalscope.service.blueprints.eval import _build_result_table


class _StubAdapter:

    name = 'gsm8k'
    primary_metric = MetricSelector(name='accuracy')
    aggregation = 'mean'
    pretty_name = 'GSM8K Pretty'
    description = ''
    category_map = {}


def test_service_result_table_uses_semantic_labels_and_values(tmp_path) -> None:
    report = ReportGenerator.generate_report(
        score_dict={
            'default': [AggScore(score=0.8567, metric_name='accuracy', aggregation='mean', num=100)]
        },
        model_name='test-model',
        data_adapter=_StubAdapter(),
    )
    report.to_json(str(tmp_path / 'reports' / 'test-model' / 'gsm8k.json'))

    table = _build_result_table(str(tmp_path))

    assert 'GSM8K Pretty' in table
    assert 'Accuracy ↑' in table
    assert '85.7%' in table
    assert '0.8567' not in table
