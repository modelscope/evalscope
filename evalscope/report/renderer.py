import os
import re
from html import escape
from typing import Dict, List, Optional

import pandas as pd

from evalscope.api.analysis import DiagnosisReport
from evalscope.api.analysis.taxonomy import ISSUE_NAMES
from evalscope.api.metric.semantics import MetricKind
from evalscope.constants import DEFAULT_LANGUAGE, PLOTLY_CDN_URL
from evalscope.metrics.semantics import format_metric_labels, format_metric_value
from evalscope.metrics.semantics.ranking import bounded_quality_ratio
from evalscope.report.combinator import get_report_list
from evalscope.report.diagnosis import load_diagnosis
from evalscope.report.report import Report, ReportKey
from evalscope.report.visualization import (
    plot_single_dataset_scores,
    plot_single_report_scores,
    plot_single_report_sunburst,
)
from evalscope.utils.io_utils import current_time
from evalscope.utils.logger import get_logger
from evalscope.utils.resource_utils import load_benchmark_data

logger = get_logger()

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'template')
_PLOTLY_CDN_CONFIG = {'responsive': True}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_diagnostic_facts(report: DiagnosisReport) -> str:
    def label(en: str, zh: str) -> str:
        return (
            f'<span class="desc-lang-content" data-desc-lang="en">{escape(en)}</span>'
            f'<span class="desc-lang-content" data-desc-lang="zh" style="display:none">{escape(zh)}</span>'
        )

    c = report.coverage
    outcome = (
        f'{label("Task failures", "任务未通过")}: {c.task_failed}/{c.task_success_known}'
        if c.task_success_known
        else label('Task outcomes unavailable', '任务成败暂无法判定')
    )
    parts = [
        f'<h3>{label("Diagnostic facts", "诊断统计")}</h3>',
        f'<p>{label("Analyzed samples", "已分析样本")}: {c.samples} · {outcome} · '
        f'{label("Unknown outcomes", "成败未知")}: {c.outcome_unknown} · '
        f'{label("Excluded scores", "未计入评分")}: {c.excluded} · '
        f'{label("Analysis calls", "分析调用")}: {report.cost.calls}</p>',
    ]
    if report.level != 'facts':
        parts.append(
            f'<p>{label("Referenced model proposals", "带引用的模型分析")}: '
            f'{c.diagnosed_ok}/{c.selected} · {label("Review population", "待排查样本")}: {c.diagnosis_population}</p>'
        )
        parts.append(
            f'<p>{label("References are locatable; conclusions await review.", "引用可定位，分析结论仍待复核。")}</p>'
        )
    prefix = f'diagnosis-{report.benchmark}-'
    reference_names = {fact.id: fact.title for fact in report.facts}
    if report.narrative:
        parts.append(f'<h4>{label("Model summary and suggestions", "模型总结与建议")}</h4><ul>')
        for item in report.narrative.findings + report.narrative.recommendations:
            if not item.refs or any(
                ref not in reference_names and ref not in report.narrative_samples for ref in item.refs
            ):
                continue
            links = ' '.join(
                escape(f'[{report.narrative_samples[ref].sample.subset}:{report.narrative_samples[ref].sample.index}]')
                if ref in report.narrative_samples
                else f'<a href="#{escape(prefix + ref, quote=True)}">[{escape(reference_names[ref])}]</a>'
                for ref in item.refs
            )
            parts.append(f'<li>{escape(item.text)} {links}</li>')
        parts.append('</ul>')
    if report.patterns:
        columns = [
            ('Issue type', '问题类型'),
            ('Failed samples', '未通过样本'),
            ('Unscored samples', '无法评分样本'),
            ('Examples', '代表样本'),
        ]
        parts.append(
            '<div class="table-wrap"><table><thead><tr>'
            + ''.join(f'<th>{label(en, zh)}</th>' for en, zh in columns)
            + '</tr></thead><tbody>'
        )
        for pattern in report.patterns[:10]:
            name = label(*ISSUE_NAMES.get(pattern.failure_type, ISSUE_NAMES['other']))
            examples = escape(', '.join(f'{sample.subset}:{sample.index}' for sample in pattern.examples))
            parts.append(
                f'<tr><td>{name}</td><td>{pattern.failed}</td><td>{pattern.unscored}</td><td>{examples}</td></tr>'
            )
        parts.append('</tbody></table></div>')
        if len(report.patterns) > 1:
            parts.append(
                f'<p>{label("A sample may appear in more than one group.", "同一样本可能出现在多个问题组。")}</p>'
            )
    cited_facts = (
        {ref for item in report.narrative.findings + report.narrative.recommendations for ref in item.refs}
        if report.narrative
        else set()
    )
    warnings = [fact for fact in report.facts if fact.severity in ('warn', 'critical') or fact.id in cited_facts]
    if warnings:
        parts.append('<ul>')
        for fact in warnings:
            value = '' if fact.value is None else f'{fact.value:.1%}' if fact.unit == 'fraction' else f'{fact.value:g}'
            parts.append(
                f'<li id="{escape(prefix + fact.id, quote=True)}"><span title="{escape(fact.definition, quote=True)}">{escape(fact.title)}</span>: {value}</li>'
            )
        parts.append('</ul>')
    return ''.join(parts)


def _process_readme_content(content: str) -> str:
    """Strip the H1 heading and the last H2 section from README markdown."""
    lines = content.splitlines()

    # Remove the first H1 heading line
    for i, line in enumerate(lines):
        if re.match(r'^#\s', line):
            lines.pop(i)
            break
        elif line.strip():  # Non-blank, non-H1 line encountered first – stop
            break

    # Remove the last H2 section (heading + everything after it)
    last_h2 = None
    for i, line in enumerate(lines):
        if re.match(r'^##\s', line):
            last_h2 = i
    if last_h2 is not None:
        lines = lines[:last_h2]

    return '\n'.join(lines).strip()


def _load_meta_readme(ds: str, lang: str = DEFAULT_LANGUAGE) -> str:
    """Load README from *_meta/{ds}.json*, select the requested language variant,
    and process the content (strip H1 + last H2 section).

    Returns an empty string when the meta file does not exist, has no readme,
    or does not contain the requested language variant.  No cross-language
    fallback is performed so callers can distinguish "has content" from
    "no content for this language".
    """
    try:
        entry = load_benchmark_data(ds).get(ds, {})
    except Exception:
        return ''
    readme = entry.get('readme', {})
    content = readme.get(lang, '')
    return _process_readme_content(content) if content else ''


def _safe_filename(name: str) -> str:
    """Strip unsafe characters from *name* for use as a file name component."""
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)


def _md_to_html(text: str) -> str:
    """Convert Markdown text to HTML for display in the report.

    Uses the ``markdown`` package (with ``extra`` extensions for tables and
    fenced code blocks).  If the package is unavailable the text is returned
    HTML-escaped inside a ``<p>`` element so the report still renders safely.
    """
    if not text:
        return ''

    import markdown as md_lib

    return md_lib.markdown(text, extensions=['extra'])


def _build_dataset_info(report_list: List[Report]) -> Dict[str, dict]:
    """Group reports by dataset, collecting model reports for each."""
    dataset_info: Dict[str, dict] = {}
    for report in report_list:
        ds = report.dataset_name
        if ds not in dataset_info:
            # Load both language variants from _meta so the HTML report can
            # switch between them when the user toggles EN / 中文.
            readme_en = _load_meta_readme(ds, lang='en')
            readme_zh = _load_meta_readme(ds, lang='zh')
            # Fall back to the description embedded in the report when no meta
            # file is available for either language.
            if not readme_en and not readme_zh:
                fallback = (report.dataset_description or '').strip()
                readme_en = fallback
            dataset_info[ds] = {
                'pretty_name': report.dataset_pretty_name or ds,
                'description_en': _md_to_html(readme_en),
                'description_zh': _md_to_html(readme_zh),
                'model_reports': {},
            }
        dataset_info[ds]['model_reports'][report.model_name] = report
    return dataset_info


def _sunburst_chart_div(report_list: List) -> str:
    """Return a Plotly HTML div for the overview sunburst chart, or empty string."""
    if not report_list:
        return ''
    try:
        fig = plot_single_report_sunburst(report_list)
    except Exception:
        logger.debug('Sunburst chart generation failed', exc_info=True)
        return ''
    if fig is None:
        return ''
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CDN_CONFIG)


def _overview_chart_div(rows: List[dict]) -> str:
    """Return a Plotly HTML div for the overview bar chart, or empty string."""
    if not rows:
        return ''
    overview_df = pd.DataFrame.from_records(rows)
    fig = plot_single_report_scores(overview_df)
    if fig is None:
        return ''
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CDN_CONFIG)


def _subset_chart_div(
    ds: str,
    model: str,
    subset_rows: List[dict],
) -> str:
    """Return a Plotly HTML div for a per-dataset subset bar chart, or empty string."""
    if not subset_rows:
        return ''
    chart_rows = [row for row in subset_rows if row['quality_score'] is not None]
    if not chart_rows:
        return ''
    subset_df = pd.DataFrame(
        {
            ReportKey.subset_name: [row['subset'] for row in chart_rows],
            ReportKey.metric_name: [row['metric'] for row in chart_rows],
            ReportKey.score: [row['quality_score'] for row in chart_rows],
            ReportKey.raw_score: [row['score'] for row in chart_rows],
            ReportKey.display_score: [row['score_display'] for row in chart_rows],
        }
    )
    fig = plot_single_dataset_scores(subset_df)
    if fig is None:
        return ''
    div_id = f'chart-{_safe_filename(ds)}-{_safe_filename(model)}'
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CDN_CONFIG, div_id=div_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gen_html_report_file(
    reports_dir: str,
    output_html_name: str = 'report.html',
) -> str:
    """Generate a self-contained interactive HTML evaluation report.

    The report is organised by dataset and contains:

    1. Title & metadata (models, datasets, timestamp)
    2. **Overview** – score-summary table + overall-score bar chart
    3. **Results by dataset** – description, subset-score table and
       subset-score bar chart per dataset (collapsible accordion)
    4. Footer with EvalScope branding

    Charts are Plotly interactive figures (hover, zoom, pan) rendered
    client-side via the Plotly.js CDN.  The CSS lives in a separate
    template file and is inlined into the output, producing a single
    fully self-contained HTML file with no external asset dependencies
    beyond the Plotly CDN.

    Args:
        reports_dir:      Root directory containing per-dataset JSON report
                          files (may be nested by model sub-directory).
        output_html_name: Name of the HTML file written into *reports_dir*.

    Returns:
        Absolute path to the generated HTML file.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as exc:
        raise ImportError('jinja2 is required to generate HTML reports: pip install jinja2') from exc

    reports_dir = os.path.abspath(reports_dir)
    if not os.path.isdir(reports_dir):
        raise ValueError(f'reports_dir does not exist or is not a directory: {reports_dir}')

    report_list: List[Report] = get_report_list([reports_dir])
    if not report_list:
        logger.warning(f'No reports found under {reports_dir}. Generating an empty report.')

    dataset_info = _build_dataset_info(report_list)
    all_models: List[str] = sorted({r.model_name for r in report_list})
    all_datasets: List[str] = sorted(dataset_info.keys())
    multi_model = len(all_models) > 1

    # ------------------------------------------------------------------
    # Overview data
    # ------------------------------------------------------------------
    summary_rows: List[dict] = []
    overview_chart_rows: List[dict] = []

    for ds in all_datasets:
        info = dataset_info[ds]
        pretty = info['pretty_name']
        for model in all_models:
            rpt: Optional[Report] = info['model_reports'].get(model)
            if rpt is None:
                continue
            primary_metric = rpt.primary_metric
            metric_labels = format_metric_labels((metric.identity, metric.semantics) for metric in rpt.metrics)
            semantics = primary_metric.semantics if primary_metric else None
            score = primary_metric.score if primary_metric else None
            summary_rows.append(
                dict(
                    dataset=pretty,
                    model=model,
                    metric=metric_labels.get(primary_metric.name, primary_metric.name) if primary_metric else 'N/A',
                    metric_raw=primary_metric.name if primary_metric else 'N/A',
                    score=score,
                    score_display=format_metric_value(score, semantics),
                    num=primary_metric.num if primary_metric else 0,
                    num_correct=primary_metric.num_correct if primary_metric else None,
                )
            )
        first: Optional[Report] = next(iter(info['model_reports'].values()), None)
        if first is not None and first.primary_metric is not None:
            metric = first.primary_metric
            metric_labels = format_metric_labels((item.identity, item.semantics) for item in first.metrics)
            quality_score = bounded_quality_ratio(metric.score, metric.semantics)
            if quality_score is not None:
                overview_chart_rows.append(
                    {
                        ReportKey.dataset_name: pretty,
                        ReportKey.metric_name: metric_labels.get(metric.name, metric.name),
                        ReportKey.score: quality_score,
                        ReportKey.raw_score: metric.score,
                        ReportKey.display_score: format_metric_value(metric.score, metric.semantics),
                    }
                )

    overview_chart_div = _overview_chart_div(overview_chart_rows)
    sunburst_chart_div = _sunburst_chart_div(report_list)

    # ------------------------------------------------------------------
    # Per-dataset sections
    # ------------------------------------------------------------------
    dataset_sections: List[dict] = []

    for ds in all_datasets:
        info = dataset_info[ds]
        pretty = info['pretty_name']
        model_sections: List[dict] = []
        overall_score: Optional[float] = None
        overall_score_display = format_metric_value(overall_score, None)

        for model in all_models:
            rpt = info['model_reports'].get(model)
            if rpt is None:
                continue
            primary_metric = rpt.primary_metric
            overall_score = primary_metric.score if primary_metric else None
            overall_score_display = format_metric_value(
                primary_metric.score if primary_metric else None,
                primary_metric.semantics if primary_metric else None,
            )

            subset_rows: List[dict] = []
            diagnostic_rows: List[dict] = []
            metric_labels = format_metric_labels((metric.identity, metric.semantics) for metric in rpt.metrics)

            for metric in rpt.metrics:
                semantics = metric.semantics
                is_diagnostic = semantics is not None and semantics.kind is MetricKind.DIAGNOSTIC
                for cat in metric.categories:
                    # Category name is stored as a tuple; join for display
                    cat_display = ' / '.join(cat.name) if cat.name else ''
                    for sub in cat.subsets:
                        if sub.name == ReportKey.overall_score:
                            continue
                        row = dict(
                            subset=sub.name,
                            category=cat_display,
                            metric=metric_labels.get(metric.name, metric.name),
                            metric_raw=metric.name,
                            score=sub.score,
                            quality_score=bounded_quality_ratio(sub.score, semantics),
                            score_display=format_metric_value(sub.score, semantics),
                            num=sub.num,
                            num_correct=sub.num_correct,
                        )
                        # Diagnostics are shown without a verdict, in their own collapsed panel.
                        (diagnostic_rows if is_diagnostic else subset_rows).append(row)

            show_category = True

            chart_div = _subset_chart_div(
                ds=ds,
                model=model,
                subset_rows=subset_rows,
            )
            analysis_raw = rpt.analysis if rpt.analysis and rpt.analysis.strip() not in ('', 'N/A') else ''
            analysis_html = _md_to_html(analysis_raw) if analysis_raw else ''
            diagnosis_status, diagnosis = load_diagnosis(str(os.path.dirname(reports_dir)), model, ds)
            if diagnosis_status == 'ready' and diagnosis is not None:
                analysis_html = _render_diagnostic_facts(diagnosis)
            elif diagnosis_status in ('stale', 'error'):
                analysis_html = '<p>Diagnosis is stale or unavailable. Re-run evalscope analyze --level facts.</p>'
            model_sections.append(
                dict(
                    model_name=model,
                    subset_rows=subset_rows,
                    diagnostic_rows=diagnostic_rows,
                    show_category=show_category,
                    chart_div=chart_div,
                    analysis_html=analysis_html,
                )
            )

        dataset_sections.append(
            dict(
                pretty_name=pretty,
                description_en=info['description_en'],
                description_zh=info['description_zh'],
                overall_score=overall_score,
                overall_score_display=overall_score_display,
                model_sections=model_sections,
                multi_model=multi_model,
            )
        )

    # ------------------------------------------------------------------
    # Render template
    # ------------------------------------------------------------------
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=False)
    template = env.get_template('report.html.j2')
    html_content = template.render(
        models=all_models,
        datasets=all_datasets,
        generated_at=current_time().strftime('%Y-%m-%d %H:%M:%S'),
        summary_rows=summary_rows,
        overview_chart_div=overview_chart_div,
        sunburst_chart_div=sunburst_chart_div,
        dataset_sections=dataset_sections,
        default_lang=DEFAULT_LANGUAGE,
        plotly_cdn_url=PLOTLY_CDN_URL,
    )

    out_path = os.path.join(reports_dir, output_html_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return out_path
