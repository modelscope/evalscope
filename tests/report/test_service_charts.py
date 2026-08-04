import plotly.graph_objects as go

from evalscope.report import Category, Metric, Report, Subset
from evalscope.service.blueprints.reports import _apply_chart_theme, _build_report_meta, _report_to_service_dict


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


def test_build_report_meta_exposes_primary_metric_name(monkeypatch) -> None:
    report = Report(
        dataset_name='throughput_suite',
        model_name='test-model',
        metrics=[
            Metric(
                name='AverageOutputTps',
                categories=[Category(name=('default', ), subsets=[Subset(name='main', score=512.0, num=1)])],
            )
        ],
    )
    monkeypatch.setattr(
        'evalscope.service.blueprints.reports.load_single_report',
        lambda _root, _name: ([report], ['throughput_suite'], {}),
    )

    metadata = _build_report_meta('run', '/tmp')

    assert metadata['metric_name'] == 'AverageOutputTps'
    assert metadata['score'] == 512.0
    assert metadata['dataset_scores'] == {'throughput_suite': 512.0}


def test_build_report_meta_prefers_explicit_overall_metric(monkeypatch) -> None:
    report = Report(
        dataset_name='document_suite',
        model_name='test-model',
        metrics=[
            Metric(
                name='text_edit',
                categories=[Category(name=('default', ), subsets=[Subset(name='main', score=0.1, num=2)])],
            ),
            Metric(
                name='overall',
                categories=[Category(name=('default', ), subsets=[Subset(name='main', score=0.9, num=2)])],
            ),
        ],
    )
    monkeypatch.setattr(
        'evalscope.service.blueprints.reports.load_single_report',
        lambda _root, _name: ([report], ['document_suite'], {}),
    )

    metadata = _build_report_meta('run', '/tmp')

    assert report.score == 0.1
    assert metadata['metric_name'] == 'overall'
    assert metadata['score'] == 0.9
    assert metadata['dataset_scores'] == {'document_suite': 0.9}
    assert _report_to_service_dict(report)['score'] == 0.9
