"""The service dataframe endpoint shares the display contract for the Correct column: an all-empty
column is dropped, and rows without a count in a mixed frame carry '-'."""
import pytest

pytest.importorskip('flask')

from types import SimpleNamespace

from evalscope.api.metric import MetricSelector, SampleScore, Score
from evalscope.metrics.aggregators import Mean
from evalscope.report.generator import ReportGenerator
from evalscope.service.app import create_app


def _write_report(root, dataset, subset_values, primary='accuracy', legacy=False):
    subset_scores = {
        subset: Mean()([SampleScore(score=Score(value=v), sample_id=f'{subset}_{i}') for i, v in enumerate(values)])
        for subset, values in subset_values.items()
    }
    adapter = SimpleNamespace(
        name=dataset, pretty_name=dataset, description='', category_map={},
        primary_metric=MetricSelector(name=primary),
    )
    data = ReportGenerator.generate_report(subset_scores, 'demo-model', adapter).to_dict()
    if legacy:
        for metric in data['metrics']:
            metric.pop('num_correct', None)
            for cat in metric['categories']:
                cat.pop('num_correct', None)
                for sub in cat['subsets']:
                    sub.pop('num_correct', None)
    import json
    report_dir = root / 'run' / 'reports' / 'demo-model'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f'{dataset}.json').write_text(json.dumps(data))


def _client(tmp_path):
    configs = tmp_path / 'run' / 'configs'
    configs.mkdir(parents=True, exist_ok=True)
    (configs / 'task_config.yaml').write_text('model: demo-model\n')
    return create_app(outputs=str(tmp_path)).test_client()


def _table(client, dataset):
    response = client.get(f'/api/v1/reports/runs/run/models/demo-model/table?view=dataset&dataset_name={dataset}')
    assert response.status_code == 200
    return response.get_json()


def test_mixed_frame_dashes_rows_without_a_count(tmp_path):
    _write_report(tmp_path, 'demo_acc', {
        'A': [{'accuracy': 1.0, 'f1': 0.5}, {'accuracy': 0.0, 'f1': 1.0}, {'accuracy': 1.0, 'f1': 0.0}],
        'B': [{'accuracy': 0.0, 'f1': 0.25}, {'accuracy': 0.0, 'f1': 0.75}],
    })
    frame = _table(_client(tmp_path), 'demo_acc')

    assert 'Correct' in frame['columns']
    counts = {(row['Metric'], row['Subset']): row['Correct'] for row in frame['data']}
    assert counts[('accuracy:mean', 'A')] == 2
    assert counts[('accuracy:mean', 'B')] == 0
    assert counts[('f1:mean', 'A')] == '-'
    assert counts[('f1:mean', 'B')] == '-'


def test_all_empty_column_is_dropped(tmp_path):
    _write_report(tmp_path, 'demo_f1', {'A': [{'f1': 1.0}, {'f1': 0.5}]}, primary='f1')
    frame = _table(_client(tmp_path), 'demo_f1')

    assert 'Correct' not in frame['columns']


def test_legacy_report_without_the_field_hides_the_column(tmp_path):
    _write_report(tmp_path, 'demo_legacy', {
        'A': [{'accuracy': 1.0}, {'accuracy': 0.0}, {'accuracy': 1.0}],
        'B': [{'accuracy': 0.0}, {'accuracy': 0.0}],
    }, legacy=True)
    frame = _table(_client(tmp_path), 'demo_legacy')

    assert 'Correct' not in frame['columns']
