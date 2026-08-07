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


def _hydrated(report: Report) -> Report:
    """Round-trip a report through the read path so its metric semantics are resolved.

    ``Report(...)`` leaves ``Metric.semantics`` unset, and the primary metric is chosen from the
    resolved role rather than from metric order. Production only ever reads a report through
    ``from_dict``, so a test that constructs one directly would assert against a state that
    cannot occur.
    """
    return Report.from_dict(report.to_dict())


def test_build_report_meta_exposes_primary_metric_name(monkeypatch) -> None:
    report = _hydrated(
        Report(
            dataset_name='throughput_suite',
            model_name='test-model',
            metrics=[
                Metric(
                    name='AverageOutputTps',
                    categories=[Category(name=('default', ), subsets=[Subset(name='main', score=512.0, num=1)])],
                )
            ],
        )
    )
    monkeypatch.setattr(
        'evalscope.service.blueprints.reports.load_single_report',
        lambda _root, _name: ([report], ['throughput_suite'], {}),
    )

    metadata = _build_report_meta('run', '/tmp')

    assert metadata['metric_name'] == 'AverageOutputTps'
    assert metadata['score'] == 512.0
    assert metadata['dataset_scores'] == {'throughput_suite': 512.0}


def test_build_report_meta_picks_the_primary_role_not_the_first_metric(monkeypatch) -> None:
    """The conclusion comes from the declared role, not from the order metrics happen to be in.

    ``text_edit`` is listed first and is what the deprecated ``score`` field still reports, but
    ``overall`` is the metric declared to carry the conclusion, so that is what the API exposes.
    """
    report = _hydrated(
        Report(
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
    )
    monkeypatch.setattr(
        'evalscope.service.blueprints.reports.load_single_report',
        lambda _root, _name: ([report], ['document_suite'], {}),
    )

    metadata = _build_report_meta('run', '/tmp')

    # The deprecated field keeps its historical first-metric value for old clients.
    assert report.score == 0.1
    assert metadata['metric_name'] == 'overall'
    assert metadata['score'] == 0.9
    assert metadata['dataset_scores'] == {'document_suite': 0.9}
    assert _report_to_service_dict(report)['score'] == 0.9
