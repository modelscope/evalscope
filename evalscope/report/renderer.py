import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from evalscope.app.utils.visualization import plot_single_dataset_scores, plot_single_report_scores
from evalscope.report.combinator import get_report_list
from evalscope.report.report import Report, ReportKey
from evalscope.utils.logger import get_logger

logger = get_logger()

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'template')
_PLOTLY_CDN_CONFIG = {'responsive': True}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
            dataset_info[ds] = {
                'pretty_name': report.dataset_pretty_name or ds,
                'description': _md_to_html((report.dataset_description or '').strip()),
                'model_reports': {},
            }
        dataset_info[ds]['model_reports'][report.model_name] = report
    return dataset_info


def _overview_chart_div(labels: List[str], scores: List[float]) -> str:
    """Return a Plotly HTML div for the overview bar chart, or empty string."""
    if not labels:
        return ''
    overview_df = pd.DataFrame({
        ReportKey.dataset_name: labels,
        ReportKey.score: scores,
    })
    fig = plot_single_report_scores(overview_df)
    if fig is None:
        return ''
    fig.update_layout(title='Overall Score by Dataset', height=400)
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CDN_CONFIG)


def _subset_chart_div(
    ds: str,
    model: str,
    pretty: str,
    metric_name: str,
    subset_labels: List[str],
    subset_scores: List[float],
    multi_model: bool,
) -> str:
    """Return a Plotly HTML div for a per-dataset subset bar chart, or empty string."""
    if not subset_labels:
        return ''
    subset_df = pd.DataFrame({
        ReportKey.subset_name: subset_labels,
        ReportKey.metric_name: [metric_name] * len(subset_labels),
        ReportKey.score: subset_scores,
    })
    fig = plot_single_dataset_scores(subset_df)
    if fig is None:
        return ''
    chart_title = f'{pretty} \u2013 Subset Scores' + (f' ({model})' if multi_model else '')
    div_id = f'chart-{_safe_filename(ds)}-{_safe_filename(model)}'
    fig.update_layout(title=chart_title, height=380)
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
    overview_labels: List[str] = []
    overview_scores: List[float] = []

    for ds in all_datasets:
        info = dataset_info[ds]
        pretty = info['pretty_name']
        for model in all_models:
            rpt: Optional[Report] = info['model_reports'].get(model)
            if rpt is None:
                continue
            main_metric_name = rpt.metrics[0].name if rpt.metrics else 'N/A'
            num = sum(cat.num for cat in rpt.metrics[0].categories) if rpt.metrics else 0
            summary_rows.append(dict(dataset=pretty, model=model, metric=main_metric_name, score=rpt.score, num=num))
        first: Optional[Report] = next(iter(info['model_reports'].values()), None)
        if first is not None:
            overview_labels.append(pretty)
            overview_scores.append(first.score)

    overview_chart_div = _overview_chart_div(overview_labels, overview_scores)

    # ------------------------------------------------------------------
    # Per-dataset sections
    # ------------------------------------------------------------------
    dataset_sections: List[dict] = []

    for ds in all_datasets:
        info = dataset_info[ds]
        pretty = info['pretty_name']
        model_sections: List[dict] = []
        overall_score = 0.0

        for model in all_models:
            rpt = info['model_reports'].get(model)
            if rpt is None:
                continue
            overall_score = rpt.score

            if not rpt.metrics:
                model_sections.append(dict(model_name=model, subset_rows=[], chart_div=''))
                continue

            main_metric = rpt.metrics[0]
            subset_rows: List[dict] = []
            subset_labels: List[str] = []
            subset_scores: List[float] = []

            for cat in main_metric.categories:
                for sub in cat.subsets:
                    if sub.name == ReportKey.overall_score:
                        continue
                    subset_rows.append(dict(subset=sub.name, metric=main_metric.name, score=sub.score, num=sub.num))
                    subset_labels.append(sub.name)
                    subset_scores.append(sub.score)

            chart_div = _subset_chart_div(
                ds=ds,
                model=model,
                pretty=pretty,
                metric_name=main_metric.name,
                subset_labels=subset_labels,
                subset_scores=subset_scores,
                multi_model=multi_model,
            )
            model_sections.append(dict(model_name=model, subset_rows=subset_rows, chart_div=chart_div))

        dataset_sections.append(
            dict(
                pretty_name=pretty,
                description=info['description'],
                overall_score=overall_score,
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
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        summary_rows=summary_rows,
        overview_chart_div=overview_chart_div,
        dataset_sections=dataset_sections,
    )

    out_path = os.path.join(reports_dir, output_html_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f'HTML report generated: {out_path}')
    return out_path
