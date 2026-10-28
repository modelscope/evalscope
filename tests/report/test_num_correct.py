"""A report states how many samples were answered correctly, not only the rounded accuracy. The count
exists only for binary metrics where higher is better; anything else leaves it unset."""
from types import SimpleNamespace
from typing import Dict, List

from evalscope.api.metric import MetricSelector, SampleScore, Score
from evalscope.metrics.aggregators import Mean
from evalscope.report import Report, gen_table
from evalscope.report.generator import ReportGenerator


def make_report(subset_values: Dict[str, List[Dict[str, float]]], primary: str) -> Report:
    subset_scores = {
        subset: Mean()([SampleScore(score=Score(value=v), sample_id=f'{subset}_{i}') for i, v in enumerate(values)])
        for subset, values in subset_values.items()
    }
    adapter = SimpleNamespace(
        name='mmmu',
        pretty_name='MMMU',
        description='',
        category_map={},
        primary_metric=MetricSelector(name=primary),
    )
    return ReportGenerator.generate_report(subset_scores, 'model', adapter)


def aggregate(metric: str, values: List[float]):
    return Mean()([SampleScore(score=Score(value={metric: v}), sample_id=str(i)) for i, v in enumerate(values)])[0]


def test_mean_counts_only_binary_values():
    assert aggregate('accuracy', [1, 0.0, True]).num_correct == 2
    assert aggregate('f1', [1.0, 0.5]).num_correct is None


def test_counts_roll_up_and_show_in_table():
    art = [{'accuracy': 1.0}, {'accuracy': 1.0}, {'accuracy': 0.0}]
    math = [{'accuracy': 0.0}, {'accuracy': 1.0}]
    report = make_report({'Art': art, 'Math': math}, 'accuracy')
    metric = report.metrics[0]

    assert [s.num_correct for s in metric.categories[0].subsets] == [2, 1]
    assert metric.categories[0].num_correct == 3
    assert metric.num_correct == 3

    table = gen_table(report_list=[report], add_overall_metric=True)
    assert 'Correct' in table
    overall = next(line for line in table.splitlines() if 'OVERALL' in line)
    assert [cell.strip() for cell in overall.split('│')][-4:-1] == ['5', '3', '60%']


def test_graded_and_lower_is_better_metrics_have_no_count():
    graded = make_report({'Art': [{'f1': 1.0}, {'f1': 0.0}], 'Math': [{'f1': 1.0}, {'f1': 0.5}]}, 'f1')
    error_rate = make_report({'Art': [{'wer': 1.0}, {'wer': 0.0}]}, 'wer')

    assert [s.num_correct for s in graded.metrics[0].categories[0].subsets] == [None, None]
    assert error_rate.metrics[0].num_correct is None
    assert 'Correct' not in gen_table(report_list=[graded])


def test_report_saved_before_the_field_still_loads():
    data = make_report({'Art': [{'accuracy': 1.0}, {'accuracy': 0.0}]}, 'accuracy').to_dict()
    metric = data['metrics'][0]
    metric.pop('num_correct')
    for category in metric['categories']:
        category.pop('num_correct')
        for subset in category['subsets']:
            subset.pop('num_correct')

    report = Report.from_dict(data)

    assert report.metrics[0].num_correct is None
    assert 'Correct' not in gen_table(report_list=[report])
