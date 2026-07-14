"""Generate an interactive HTML performance benchmark report.

This module is the public entry point for HTML report generation.
It delegates data loading to ``perf_data.RunLoader``, chart construction
to ``perf_charts``, and final rendering to a Jinja2 template.

Public API
----------
    gen_perf_html_report(output_dir, results, args, output_html_name) -> str
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from evalscope.constants import DEFAULT_LANGUAGE, PLOTLY_CDN_URL
from evalscope.utils.io_utils import current_time
from evalscope.utils.logger import get_logger
from evalscope.version import __version__ as _evalscope_version
from .summary import (
    build_basic_info,
    build_best_config,
    build_recommendations,
    build_summary_items,
    build_summary_table,
    is_embedding,
)

logger = get_logger()

_TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    'report',
    'template',
)

# ---------------------------------------------------------------------------
# Per-run section builder
# ---------------------------------------------------------------------------


def _build_run_section(run, is_embedding_flag: bool) -> Dict[str, Any]:
    """Build the template-ready dict for one per-run accordion card."""
    from . import perf_charts as charts

    pct = run.percentiles.to_list()
    pct_charts = charts.build_percentile_chart(run, is_embedding_flag)
    return {
        'name': run.name,
        'total_requests': run.summary.total_requests,
        # benchmark_summary.json displayed as a stat grid
        'summary_items': build_summary_items(run.summary, is_embedding_flag),
        # percentile table
        'percentile_columns': list(pct[0].keys()) if pct else [],
        'percentile_rows': [list(p.values()) for p in pct],
        # charts
        'percentile_chart': pct_charts[0],
        'percentile_token_lat_chart': pct_charts[1],
        'request_detail_tabs': charts.build_request_detail_tabs(run, is_embedding_flag),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gen_perf_html_report(
    output_dir: str,
    results: dict,
    args: Any,
    output_html_name: str = 'perf_report.html',
) -> str:
    """Generate a self-contained interactive HTML performance benchmark report.

    Args:
        output_dir:       Root output directory (contains ``parallel_*`` subdirs).
        results:          Dict returned by ``run_multi_benchmark`` (may be empty).
        args:             Arguments namespace with benchmark configuration.
        output_html_name: Output filename (default: ``perf_report.html``).

    Returns:
        Absolute path to the generated HTML file, or ``''`` on failure.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as exc:
        raise ImportError('jinja2 is required: pip install jinja2') from exc

    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError('plotly is required: pip install plotly') from exc

    from . import perf_charts as charts
    from .perf_data import RunLoader

    output_dir = os.path.abspath(output_dir)
    runs = RunLoader.load_all(output_dir)

    if not runs:
        logger.warning(f'No benchmark runs found under {output_dir}. Skipping HTML report.')
        return ''

    # Resolve API type from args or first run's stored args
    api_type = getattr(args, 'api', '') or runs[0].args.get('api', '')
    is_emb = is_embedding(api_type)
    first_args = runs[0].args

    # ── Tab groups ──────────────────────────────────────────────────────────
    latency_tabs = [{'label': 'Latency', 'chart': charts.build_latency_chart(runs)}]
    if not is_emb:
        latency_tabs.append({'label': 'TTFT', 'chart': charts.build_ttft_chart(runs)})
        latency_tabs.append({'label': 'TPOT', 'chart': charts.build_tpot_chart(runs)})

    throughput_tabs = [
        {
            'label': 'Request Throughput',
            'chart': charts.build_rps_chart(runs)
        },
        {
            'label': 'Token Throughput',
            'chart': charts.build_throughput_chart(runs, is_emb)
        },
        {
            'label': 'Success Rate',
            'chart': charts.build_success_chart(runs)
        },
    ]

    # ── Summary table ───────────────────────────────────────────────────────
    summary_columns, summary_rows = build_summary_table(runs, is_emb)

    # ── Template rendering ──────────────────────────────────────────────────
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=False)
    template = env.get_template('perf_report.html.j2')

    html_content = template.render(
        model=first_args.get('model', first_args.get('model_id', 'N/A')),
        api_type=api_type,
        dataset=first_args.get('dataset', 'N/A'),
        generated_at=current_time().strftime('%Y-%m-%d %H:%M:%S'),
        evalscope_version=_evalscope_version,
        basic_info=build_basic_info(first_args, runs, is_emb),
        summary_columns=summary_columns,
        summary_rows=summary_rows,
        best_config=build_best_config(runs),
        recommendations=build_recommendations(runs),
        latency_tabs=latency_tabs,
        throughput_tabs=throughput_tabs,
        run_sections=[_build_run_section(r, is_emb) for r in runs],
        default_lang=DEFAULT_LANGUAGE,
        plotly_cdn_url=PLOTLY_CDN_URL,
    )

    out_path = os.path.join(output_dir, output_html_name)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(html_content)

    return out_path
