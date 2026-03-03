import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional

from evalscope.report.combinator import get_report_list
from evalscope.report.report import Report, ReportKey
from evalscope.utils.logger import get_logger

# Non-interactive backend, suitable for headless API / CLI environments
matplotlib.use('Agg')

logger = get_logger()

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------
_BAR_COLOR = '#4C72B0'  # default single-series bar colour (muted blue)
_GRID_COLOR = '#EBEBEB'  # light grid lines
_SCORE_FONTSIZE = 9
_AXIS_FONTSIZE = 10
_TITLE_FONTSIZE = 12

try:
    # matplotlib >= 3.5
    _RDYLGN = matplotlib.colormaps['RdYlGn']
except Exception:
    _RDYLGN = matplotlib.cm.get_cmap('RdYlGn')  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fig_width(n: int, min_w: float = 4.0, per_item: float = 1.5, max_w: float = 14.0) -> float:
    """Return a sensible figure width (inches) for *n* bars.

    A fixed per-item scaling prevents a single bar from ballooning into a
    full-width colour block.
    """
    return max(min_w, min(per_item * n + 1.5, max_w))


def _format_markdown_table(headers: List[str], rows: List[List]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    if not headers:
        return ''
    header_line = '| ' + ' | '.join(str(h) for h in headers) + ' |'
    sep_line = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    data_lines = ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join([header_line, sep_line] + data_lines)


def _save_bar_chart(
    labels: List[str],
    scores: List[float],
    title: str,
    img_path: str,
    use_score_colors: bool = False,
) -> bool:
    """Draw a bar chart and save to *img_path*.  Returns True on success.

    Key aesthetics
    --------------
    - Bar width is **fixed at 0.4** data units so a single bar never fills the
      whole figure – it stays at 40 % of the unit interval regardless of how
      wide the figure is.
    - Figure width scales linearly with the number of bars (capped at 14 in.).
    - When *use_score_colors* is True, bars are coloured red → yellow → green
      according to their score value.
    """
    if not labels:
        return False

    n = len(labels)
    bar_width = 0.4
    fig_w = _fig_width(n)

    fig, ax = plt.subplots(figsize=(fig_w, 4.2))

    x = np.arange(n)

    if use_score_colors and n > 1:
        bar_colors = [_RDYLGN(float(s)) for s in scores]
    else:
        bar_colors = _BAR_COLOR

    bars = ax.bar(x, scores, width=bar_width, color=bar_colors, zorder=2)

    # Value annotations above each bar
    for bar, score in zip(bars, scores):
        try:
            s = float(score)
        except Exception:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            s + 0.015,
            f'{s:.3f}',
            ha='center',
            va='bottom',
            fontsize=_SCORE_FONTSIZE,
            fontweight='bold',
            color='#333333',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=20 if n > 5 else 0,
        ha='right' if n > 5 else 'center',
        fontsize=_AXIS_FONTSIZE,
    )
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel('Score', fontsize=_AXIS_FONTSIZE)
    ax.set_title(title, fontsize=_TITLE_FONTSIZE, pad=10)
    ax.yaxis.grid(True, color=_GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path, dpi=130)
    plt.close(fig)
    return True


def _safe_filename(name: str) -> str:
    """Strip unsafe characters from *name* for use as a file name component."""
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gen_markdown_report(
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

    Images are saved under ``reports_dir/_assets/`` (by default) using
    relative paths so that the Markdown file and its assets can be zipped
    and opened anywhere.

    Args:
        reports_dir:     Root directory that contains the per-dataset JSON
                         report files (may be nested by model sub-directory).
        output_md_name:  Name of the Markdown file written into *reports_dir*.
        assets_dir_name: Name of the sub-directory for image assets.

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
        f'- **Generated at:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC',
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
        img_path = os.path.join(assets_dir, 'overview_scores.png')
        ok = _save_bar_chart(
            labels=overview_labels,
            scores=overview_scores,
            title='Overall Score by Dataset',
            img_path=img_path,
            use_score_colors=False,
        )
        if ok:
            rel = os.path.relpath(img_path, start=reports_dir)
            lines += [
                '### Overall Score Chart',
                '',
                f'![Overall Score by Dataset]({rel})',
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
                img_path = os.path.join(assets_dir, f'{safe_ds}_{safe_m}_subsets.png')
                chart_title = (f'{pretty} – Subset Scores ({model})' if multi_model else f'{pretty} – Subset Scores')
                ok = _save_bar_chart(
                    labels=subset_labels,
                    scores=subset_scores,
                    title=chart_title,
                    img_path=img_path,
                    use_score_colors=(len(subset_labels) > 1),
                )
                if ok:
                    rel = os.path.relpath(img_path, start=reports_dir)
                    lines += [
                        f'![{pretty} subset scores]({rel})',
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
    logger.info(f'Markdown report generated at: {md_path}')
