import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from evalscope.app.utils.visualization import plot_single_dataset_scores, plot_single_report_scores
from evalscope.report.combinator import get_report_list
from evalscope.report.report import Report, ReportKey
from evalscope.utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_markdown_table(headers: List[str], rows: List[List]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    if not headers:
        return ''
    header_line = '| ' + ' | '.join(str(h) for h in headers) + ' |'
    sep_line = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    data_lines = ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join([header_line, sep_line] + data_lines)


def _save_plotly_html(fig, html_path: str) -> bool:
    """Save a Plotly figure as a self-contained HTML file.  Returns True on success."""
    try:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path, include_plotlyjs='cdn', full_html=False)
        return True
    except Exception as e:
        logger.warning(f'Failed to save chart to {html_path}: {e}')
        return False


def _safe_filename(name: str) -> str:
    """Strip unsafe characters from *name* for use as a file name component."""
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)


def _iframe(rel_path: str, height: int = 420) -> str:
    """Return a Markdown-embeddable iframe tag for a relative HTML chart path."""
    return f'<iframe src="{rel_path}" width="100%" height="{height}" frameborder="0"></iframe>'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gen_report_file(
    reports_dir: str,
    output_md_name: str = 'report.md',
    assets_dir_name: str = '_assets',
) -> str:
    """Generate a dataset-wise Markdown evaluation report.

    The report covers **all** JSON reports found (recursively) under
    *reports_dir* and is organised by dataset.  It contains:

    1. Title & metadata (models, datasets, timestamp)
    2. **Overview** – score-summary table + overall-score bar chart
    3. **Results by dataset** – dataset description, subset-score table
       and subset-score bar chart for each dataset
    4. Footer with EvalScope branding

    Charts are saved as interactive Plotly HTML files under
    ``reports_dir/_assets/`` (by default) and embedded via ``<iframe>``
    so that the Markdown file and its assets can be opened anywhere that
    renders HTML (GitHub Pages, VS Code preview, Jupyter, etc.).

    Args:
        reports_dir:     Root directory that contains the per-dataset JSON
                         report files (may be nested by model sub-directory).
        output_md_name:  Name of the Markdown file written into *reports_dir*.
        assets_dir_name: Name of the sub-directory for chart assets.

    Returns:
        Absolute path to the generated Markdown file.
    """
    reports_dir = os.path.abspath(reports_dir)
    if not os.path.isdir(reports_dir):
        raise ValueError(f'reports_dir does not exist or is not a directory: {reports_dir}')

    report_list: List[Report] = get_report_list([reports_dir])
    if not report_list:
        logger.warning(f'No reports found under {reports_dir}. Generating an empty report.')

    assets_dir = os.path.join(reports_dir, assets_dir_name)

    # -------------------------------------------------------------------
    # Build lookup: dataset_name → { pretty_name, description,
    #                                 model_reports: {model_name: Report} }
    # -------------------------------------------------------------------
    dataset_info: Dict[str, dict] = {}
    for report in report_list:
        ds = report.dataset_name
        if ds not in dataset_info:
            dataset_info[ds] = {
                'pretty_name': report.dataset_pretty_name or ds,
                'description': (report.dataset_description or '').strip(),
                'model_reports': {},
            }
        dataset_info[ds]['model_reports'][report.model_name] = report

    all_models: List[str] = sorted({r.model_name for r in report_list})
    all_datasets: List[str] = sorted(dataset_info.keys())
    multi_model = len(all_models) > 1

    lines: List[str] = []

    # ===================================================================
    # 1. Title & metadata
    # ===================================================================
    lines += [
        '# EvalScope Evaluation Report',
        '',
        f'- **Model(s):** {", ".join(all_models) if all_models else "N/A"}',
        f'- **Datasets:** {", ".join(all_datasets) if all_datasets else "N/A"}',
        f'- **Generated at:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '- **Generated by:** [EvalScope](https://github.com/modelscope/evalscope)',
        '',
        '---',
        '',
    ]

    if not report_list:
        lines.append('No evaluation reports were found under the specified directory.')
        _flush(lines, reports_dir, output_md_name)
        return os.path.join(reports_dir, output_md_name)

    # ===================================================================
    # 2. Overview
    # ===================================================================
    lines += [
        '## Overview',
        '',
        'Summary of overall scores across all evaluated datasets.',
        '',
        '### Score Summary',
        '',
    ]

    # Summary table rows and data for the overview bar chart
    summary_table_rows: List[List] = []
    overview_labels: List[str] = []
    overview_scores: List[float] = []

    for ds in all_datasets:
        info = dataset_info[ds]
        pretty = info['pretty_name']
        for model in all_models:
            report = info['model_reports'].get(model)
            if report is None:
                continue
            main_metric_name = report.metrics[0].name if report.metrics else 'N/A'
            num = sum(cat.num for cat in report.metrics[0].categories) if report.metrics else 0
            summary_table_rows.append([pretty, model, main_metric_name, f'{report.score:.4f}', str(num)])

        # Overview chart: use first model only (or aggregate for multi-model later)
        first_report: Optional[Report] = next(iter(info['model_reports'].values()), None)
        if first_report is not None:
            overview_labels.append(pretty)
            overview_scores.append(first_report.score)

    if summary_table_rows:
        lines.append(
            _format_markdown_table(
                headers=['Dataset', 'Model', 'Metric', 'Score', 'Num'],
                rows=summary_table_rows,
            )
        )
        lines.append('')

    # Overview bar chart (overall score per dataset)
    if overview_labels:
        html_path = os.path.join(assets_dir, 'overview_scores.html')
        overview_df = pd.DataFrame({
            ReportKey.dataset_name: overview_labels,
            ReportKey.score: overview_scores,
        })
        fig = plot_single_report_scores(overview_df)
        if fig is not None:
            fig.update_layout(title='Overall Score by Dataset')
            ok = _save_plotly_html(fig, html_path)
            if ok:
                rel = os.path.relpath(html_path, start=reports_dir)
                lines += [
                    '### Overall Score Chart',
                    '',
                    _iframe(rel),
                    '',
                ]

    lines += ['---', '']

    # ===================================================================
    # 3. Results by dataset
    # ===================================================================
    lines += [
        '## Results by Dataset',
        '',
    ]

    for ds in all_datasets:
        info = dataset_info[ds]
        pretty = info['pretty_name']
        description = info['description']

        lines += [
            f'### {pretty}',
            '',
        ]

        # Dataset description (collapsed by default)
        if description:
            lines += [
                '<details>',
                '<summary>Dataset Description</summary>',
                '',
                description,
                '',
                '</details>',
                '',
            ]

        # Per-model breakdown
        for model in all_models:
            report = info['model_reports'].get(model)
            if report is None:
                continue

            if multi_model:
                lines += [f'#### Model: {model}', '']

            if not report.metrics:
                lines += ['_No metrics available._', '']
                continue

            main_metric = report.metrics[0]

            # Collect subset rows (skip the synthetic OVERALL row)
            subset_rows: List[List] = []
            subset_labels: List[str] = []
            subset_scores: List[float] = []

            for cat in main_metric.categories:
                for sub in cat.subsets:
                    if sub.name == ReportKey.overall_score:
                        continue
                    subset_rows.append([sub.name, main_metric.name, f'{sub.score:.4f}', str(sub.num)])
                    subset_labels.append(sub.name)
                    subset_scores.append(sub.score)

            # Subset table
            if subset_rows:
                lines.append(_format_markdown_table(
                    headers=['Subset', 'Metric', 'Score', 'Num'],
                    rows=subset_rows,
                ))
                lines.append('')

            # Subset bar chart
            if subset_labels:
                safe_ds = _safe_filename(ds)
                safe_m = _safe_filename(model)
                html_path = os.path.join(assets_dir, f'{safe_ds}_{safe_m}_subsets.html')
                chart_title = (f'{pretty} – Subset Scores ({model})' if multi_model else f'{pretty} – Subset Scores')
                subset_df = pd.DataFrame({
                    ReportKey.subset_name: subset_labels,
                    ReportKey.metric_name: [main_metric.name] * len(subset_labels),
                    ReportKey.score: subset_scores,
                })
                fig = plot_single_dataset_scores(subset_df)
                if fig is not None:
                    fig.update_layout(title=chart_title)
                    ok = _save_plotly_html(fig, html_path)
                    if ok:
                        rel = os.path.relpath(html_path, start=reports_dir)
                        lines += [
                            _iframe(rel),
                            '',
                        ]

        lines += ['', '---', '']

    # ===================================================================
    # 4. Footer
    # ===================================================================
    lines += [
        '_Report generated by **[EvalScope](https://github.com/modelscope/evalscope)**._',
        '',
    ]

    _flush(lines, reports_dir, output_md_name)
    return os.path.join(reports_dir, output_md_name)


def _flush(lines: List[str], reports_dir: str, output_md_name: str) -> None:
    md_path = os.path.join(reports_dir, output_md_name)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
