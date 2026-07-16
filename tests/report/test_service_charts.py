import plotly.graph_objects as go

from evalscope.service.blueprints.reports import _apply_chart_theme


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
