"""Semantic display coverage for the standalone HTML report."""

from pathlib import Path

import pytest

from evalscope.report import renderer
from evalscope.report.report import Report


def test_html_uses_semantic_labels_values_and_raw_name_tooltips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = Report.from_dict({
        'dataset_name': 'gsm8k',
        'dataset_pretty_name': 'GSM8K',
        'model_name': 'fixture-model',
        'metrics': [{
            'name': 'mean_acc',
            'categories': [{
                'name': ['default'],
                'subsets': [{
                    'name': 'test',
                    'score': 0.8567,
                    'num': 10,
                }],
            }],
        }],
    })
    monkeypatch.setattr(renderer, 'get_report_list', lambda _: [report])
    monkeypatch.setattr(renderer, '_overview_chart_div', lambda _: '')
    monkeypatch.setattr(renderer, '_sunburst_chart_div', lambda _: '')
    monkeypatch.setattr(renderer, '_subset_chart_div', lambda **_: '')

    output = renderer.gen_html_report_file(str(tmp_path))
    html = Path(output).read_text(encoding='utf-8')

    assert 'title="accuracy:mean">Accuracy ↑</span>' in html
    assert '<span class="score-pill" data-score="0.8567">85.7%</span>' in html
    assert '<td class="score-cell">85.7%</td>' in html
