from pathlib import Path

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric.semantics import MetricIdentity
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.report import Report
from evalscope.report.data_frames import get_acc_report_df


def test_grounding_comparison_reports_mean_iou_after_reload(tmp_path: Path) -> None:
    config = TaskConfig(datasets=['refcoco'], dataset_args={'refcoco': {'extra_params': {'eval_mode': 'bbox_rec'}}})
    adapter = get_benchmark('refcoco', config)
    predictions = [[0, 0, 0.5, 1], [0, 0, 1, 1], [0.5, 0, 1, 1]]
    target = [0, 0, 0.5, 1]
    scores = []
    for index, prediction in enumerate(predictions):
        sample = Sample(
            id=index, input='locate the object', target=str(target), metadata={'bbox': target, 'eval_mode': 'bbox_rec'}
        )
        state = TaskState(
            model='mock', sample=sample, output=ModelOutput.from_content('mock', str(prediction)), completed=True
        )
        scores.append(adapter.calculate_metrics(state))

    assert [score.score.value['IoU'] for score in scores] == [1.0, 0.5, 0.0]
    report = adapter.generate_report(
        {'testA': adapter.aggregate_scores(scores)}, model_name='mock', output_dir=str(tmp_path)
    )
    path = str(tmp_path / 'refcoco.json')
    report.to_json(path)
    loaded = Report.from_json(path)
    assert loaded.primary_metric_identity == MetricIdentity(name='iou', aggregation='mean')
    assert loaded.primary_metric.semantics.semantic_id == 'quality.iou.ratio'
    assert get_acc_report_df([loaded]).iloc[0][['Score', 'Num']].to_dict() == {'Score': 0.5, 'Num': 3}


@pytest.mark.parametrize(('name', 'aggregation'), [('mean_IoU', 'mean'), ('IoU', 'identity'), ('io_u', 'identity')])
def test_legacy_iou_spelling_resolves_to_grounding_primary(name: str, aggregation: str) -> None:
    report = Report.from_dict({
        'dataset_name': 'refcoco',
        'metrics': [{'name': name, 'score': 0.5, 'num': 3}],
    })
    assert report.primary_metric_identity == MetricIdentity(name='iou', aggregation=aggregation)
    assert report.score == 0.5
    assert report.primary_metric.num == 3
