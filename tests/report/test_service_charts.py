import plotly.graph_objects as go
import pytest

from evalscope.api.metric.semantics import MetricIdentity
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
from evalscope.report import Category, Metric, Report, ReportRef, Subset
from evalscope.report.visualization import plot_multi_report_radar, plot_single_report_scores
from evalscope.service.blueprints.reports import _apply_chart_theme, _build_report_meta, _report_to_service_dict
from evalscope.utils.data_utils import get_comparison_quality_report_df, get_quality_report_df


def test_apply_chart_theme_uses_light_template_for_light_console() -> None:
    fig = go.Figure()

    _apply_chart_theme(fig, 'light')

    assert fig.layout.template.layout.plot_bgcolor == 'white'
    assert fig.layout.template.layout.paper_bgcolor == 'white'


def test_apply_chart_theme_keeps_dark_template_as_safe_default() -> None:
    fig = go.Figure()

    _apply_chart_theme(fig, 'invalid')

    assert fig.layout.template.layout.plot_bgcolor == 'rgb(17,17,17)'
    assert fig.layout.template.layout.paper_bgcolor == 'rgb(17,17,17)'


def _hydrated(report: Report) -> Report:
    """Round-trip a report through the read path so its metric semantics are resolved.

    ``Report(...)`` leaves ``Metric.semantics`` unset, and the primary metric is chosen from the
    resolved role rather than from metric order. Production only ever reads a report through
    ``from_dict``, so a test that constructs one directly would assert against a state that
    cannot occur.
    """
    return Report.from_dict(report.to_dict())


def test_build_report_meta_exposes_primary_metric_identity(monkeypatch) -> None:
    identity = MetricIdentity(name='output_throughput', aggregation='mean')
    semantics = SEMANTIC_BASELINES['perf.throughput.tokens_per_second']
    report = Report(
        dataset_name='throughput_suite',
        dataset_pretty_name='Throughput Suite',
        model_name='test-model',
        metrics=[
            Metric(
                identity=identity,
                semantics=semantics,
                categories=[Category(name=('default', ), subsets=[Subset(name='main', score=512.0, num=1)])],
            )
        ],
        primary_metric_identity=identity,
    )
    monkeypatch.setattr(
        'evalscope.service.blueprints.reports.load_report_bundle',
        lambda _root, _ref: ([report], ['throughput_suite'], {}),
    )

    metadata = _build_report_meta(ReportRef(run_id='run', model_id='test-model'), '/tmp')

    assert metadata['run_id'] == 'run'
    assert metadata['model_id'] == 'test-model'
    assert metadata['primary_metrics'] == [{
        'dataset_name': 'throughput_suite',
        'dataset_pretty_name': 'Throughput Suite',
        'identity': identity.model_dump(),
        'score': 512.0,
        'semantics': semantics.model_dump(mode='json'),
    }]
    assert metadata['dataset_name'] == 'throughput_suite'
    assert metadata['dataset_pretty_name'] == 'Throughput Suite'
    assert set(('quality_ratio', '_quality_group')).isdisjoint(metadata)
    assert set(('metric_name', 'score', 'dataset_scores')).isdisjoint(metadata)


def test_build_report_meta_picks_the_primary_identity_not_the_first_metric(monkeypatch) -> None:
    """The declared structured identity, not metric order, selects the conclusion."""
    accuracy_identity = MetricIdentity(name='accuracy', aggregation='mean')
    normalized_identity = MetricIdentity(name='normalized_score', aggregation='mean')
    accuracy_semantics = SEMANTIC_BASELINES['quality.accuracy.ratio']
    normalized_semantics = SEMANTIC_BASELINES['quality.score.ratio']
    report = Report(
        dataset_name='document_suite',
        model_name='test-model',
        metrics=[
            Metric(
                identity=accuracy_identity,
                semantics=accuracy_semantics,
                categories=[Category(name=('default', ), subsets=[Subset(name='main', score=0.1, num=2)])],
            ),
            Metric(
                identity=normalized_identity,
                semantics=normalized_semantics,
                categories=[Category(name=('default', ), subsets=[Subset(name='main', score=0.9, num=2)])],
            ),
        ],
        primary_metric_identity=normalized_identity,
    )
    monkeypatch.setattr(
        'evalscope.service.blueprints.reports.load_report_bundle',
        lambda _root, _ref: ([report], ['document_suite'], {}),
    )

    metadata = _build_report_meta(ReportRef(run_id='run', model_id='test-model'), '/tmp')

    assert report.primary_metric.score == 0.9
    assert metadata['primary_metrics'][0]['identity'] == normalized_identity.model_dump()
    assert metadata['primary_metrics'][0]['score'] == 0.9
    assert set(('quality_ratio', '_quality_group')).isdisjoint(metadata)
    payload = _report_to_service_dict(report)
    assert payload['primary_metric_identity'] == normalized_identity.model_dump()
    assert set(('score', 'metric_name', 'dataset_scores')).isdisjoint(payload)


def test_build_report_meta_does_not_rank_multiple_datasets(monkeypatch) -> None:
    reports = [
        _semantic_report('accuracy', 0.8, 'quality.accuracy.ratio'),
        _semantic_report('f1', 0.6, 'quality.f1.ratio'),
    ]
    monkeypatch.setattr(
        'evalscope.service.blueprints.reports.load_report_bundle',
        lambda _root, _ref: (reports, ['accuracy', 'f1'], {}),
    )

    metadata = _build_report_meta(ReportRef(run_id='run', model_id='test-model'), '/tmp')

    assert len(metadata['primary_metrics']) == 2
    assert set(('quality_ratio', '_quality_group')).isdisjoint(metadata)


def _semantic_report(dataset_name: str, score: float, semantic_id: str) -> Report:
    semantics = SEMANTIC_BASELINES[semantic_id]
    identity = MetricIdentity(name=dataset_name, aggregation='mean')
    return Report(
        dataset_name=dataset_name,
        model_name='test-model',
        metrics=[
            Metric(
                identity=identity,
                semantics=semantics,
                categories=[Category(name=('default', ), subsets=[Subset(name='main', score=score, num=1)])],
            )
        ],
        primary_metric_identity=identity,
    )


def test_comparison_chart_uses_quality_ratio_and_keeps_native_labels() -> None:
    quality_df = get_quality_report_df([
        _semantic_report('points', 87.5, 'quality.score.points_100'),
        _semantic_report('error', 0.2, 'quality.error_rate.ratio'),
    ])

    assert quality_df['Score'].tolist() == pytest.approx([0.875, 0.8])
    assert quality_df['Display Score'].tolist() == ['87.5%', '20%']

    figure = plot_single_report_scores(quality_df)
    assert figure.layout.yaxis.range == (0, 1)
    assert list(figure.data[0].text) == ['87.5%', '20%']


def test_comparison_chart_omits_unbounded_metrics() -> None:
    quality_df = get_quality_report_df([
        _semantic_report('throughput', 512.0, 'perf.throughput.tokens_per_second'),
    ])

    assert quality_df.empty
    assert plot_single_report_scores(quality_df) is None


def test_comparison_chart_keeps_separate_runs_of_the_same_model() -> None:
    first_ref = ReportRef(run_id='20260810_100000', model_id='test-model')
    second_ref = ReportRef(run_id='20260810_110000', model_id='test-model')
    quality_df = get_comparison_quality_report_df([
        (first_ref, [_semantic_report('points', 75.0, 'quality.score.points_100')]),
        (second_ref, [_semantic_report('points', 90.0, 'quality.score.points_100')]),
    ])

    assert quality_df['Model'].tolist() == [
        'test-model (20260810_100000)',
        'test-model (20260810_110000)',
    ]
    figure = plot_multi_report_radar(quality_df)
    assert [trace.name for trace in figure.data] == [
        'test-model (20260810_100000)',
        'test-model (20260810_110000)',
    ]
