"""The interval computed by ``Mean`` must survive into the report, not only the aggregate."""

from types import SimpleNamespace

from evalscope.api.metric import MetricSelector, SampleScore, Score
from evalscope.metrics.aggregators import Mean
from evalscope.report.generator import ReportGenerator


def _report(metric: str, values: list):
    scores = Mean()([SampleScore(score=Score(value={metric: value}), sample_id=str(i))
                     for i, value in enumerate(values)])
    adapter = SimpleNamespace(
        name='mmmu',
        pretty_name='MMMU',
        description='',
        category_map={},
        primary_metric=MetricSelector(name=metric),
    )
    return ReportGenerator.generate_report({'Art': scores}, 'model', adapter)


def test_ratio_interval_reaches_the_report_payload() -> None:
    report = _report('accuracy', [1, 0, 1, 1])

    subset = report.metrics[0].categories[0].subsets[0]
    assert subset.metadata is not None
    assert subset.metadata['ci_method'] == 'wilson'
    assert subset.metadata['ci_low'] < subset.score < subset.metadata['ci_high']

    # The interval is only useful if it is serialized with the report.
    payload = report.to_dict()
    assert 'ci_low' in str(payload)
    assert 'wilson' in str(payload)


def test_continuous_metric_report_has_no_interval() -> None:
    report = _report('f1', [0.4, 0.6])

    subset = report.metrics[0].categories[0].subsets[0]
    assert subset.metadata is None
    assert 'wilson' not in str(report.to_dict())
