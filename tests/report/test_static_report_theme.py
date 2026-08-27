from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).parents[2] / 'evalscope' / 'report' / 'template'


def _render_template(name: str) -> str:
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=False)
    common = {
        'generated_at': '',
        'default_lang': 'en',
        'plotly_cdn_url': '',
    }
    if name == 'report.html.j2':
        return env.get_template(name).render(
            **common,
            models=[],
            datasets=[],
            summary_rows=[],
            overview_chart_div='',
            sunburst_chart_div='',
            dataset_sections=[],
        )
    return env.get_template(name).render(
        **common,
        model='',
        api_type='',
        dataset='',
        evalscope_version='',
        basic_info={},
        summary_columns=[],
        summary_rows=[],
        best_config={},
        recommendations=[],
        latency_tabs=[],
        throughput_tabs=[],
        run_sections=[],
    )


@pytest.mark.parametrize('template_name', ['report.html.j2', 'perf_report.html.j2'])
def test_static_report_inherits_console_theme_before_first_paint(template_name: str) -> None:
    html = _render_template(template_name)

    assert html.index("localStorage.getItem('evalscope-theme')") < html.index('<style>')
    assert html.index("document.documentElement.setAttribute('data-theme', theme)") < html.index('<body>')
    assert '[data-theme="light"]' in html
    assert '--bg:             #faf9f5;' in html


@pytest.mark.parametrize('template_name', ['report.html.j2', 'perf_report.html.j2'])
def test_static_report_applies_plotly_theme(template_name: str) -> None:
    html = _render_template(template_name)

    assert 'function applyPlotlyTheme()' in html
    assert "document.querySelectorAll('.plotly-graph-div')" in html
    assert 'window.Plotly.relayout(plot, update)' in html
    assert "styles.getPropertyValue('--plotly-text')" in html
    assert "addEventListener('storage'" not in html


def test_perf_report_uses_theme_aware_shared_surface() -> None:
    html = _render_template('perf_report.html.j2')

    assert '.tab-nav {' in html
    assert 'background: var(--surface-subtle);' in html
