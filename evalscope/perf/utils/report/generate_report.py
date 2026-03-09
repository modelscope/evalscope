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
from datetime import datetime
from typing import Any, Dict, List, Optional

from evalscope.utils.logger import get_logger
from evalscope.version import __version__ as _evalscope_version

logger = get_logger()

_TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    'report',
    'template',
)

# ---------------------------------------------------------------------------
# API-type detection
# ---------------------------------------------------------------------------


def _is_embedding(api_type: str) -> bool:
    """Return True when *api_type* indicates an embedding / rerank model."""
    lower = (api_type or '').lower()
    return 'embedding' in lower or 'rerank' in lower or 'embed' in lower


# ---------------------------------------------------------------------------
# Overview data builders
# ---------------------------------------------------------------------------


def _build_basic_info(
    args_dict: dict,
    runs: list,
    is_embedding_flag: bool,
) -> OrderedDict:
    """Produce the key/value pairs shown in the Overview › Basic Information card."""
    info: OrderedDict = OrderedDict()
    info['Model'] = args_dict.get('model', args_dict.get('model_id', 'N/A'))
    info['API Type'] = args_dict.get('api', 'N/A')
    info['Dataset'] = args_dict.get('dataset', 'N/A')

    total_req = sum(r.summary.get('Total requests', 0) for r in runs)
    succeed = sum(r.summary.get('Succeed requests', 0) for r in runs)
    total_time = sum(r.summary.get('Time taken for tests (s)', 0) for r in runs)

    info['Total Requests'] = f'{total_req:,}'
    info['Succeed Requests'] = f'{succeed:,}'
    info['Total Test Time'] = f'{total_time:.2f} s'

    if is_embedding_flag:
        total_input = sum(
            r.summary.get('Average input tokens per request', 0) * r.summary.get('Succeed requests', 0) for r in runs
        )
        info['Total Input Tokens'] = f'{total_input:,.0f}'
    else:
        total_output = sum(
            r.summary.get('Average output tokens per request', 0) * r.summary.get('Succeed requests', 0) for r in runs
        )
        info['Total Output Tokens'] = f'{total_output:,.0f}'

    return info


def _build_summary_table(runs: list, is_embedding_flag: bool):
    """Build the cross-run summary table. Returns *(columns, rows)*."""
    if is_embedding_flag:
        columns = [
            'Conc.',
            'Rate',
            'RPS',
            'Avg Lat.(s)',
            'P99 Lat.(s)',
            'Avg Inp.TPS',
            'Avg Inp.Tok',
            'Success Rate',
        ]
        rows = []
        for r in runs:
            s = r.summary
            rate = s.get('Request rate (req/s)', -1)
            rows.append([
                str(int(s.get('Number of concurrency', 0))),
                'INF' if rate == -1 else f'{rate:.2f}',
                f"{s.get('Request throughput (req/s)', 0):.4f}",
                f"{s.get('Average latency (s)', 0):.3f}",
                f"{r.get_p99('Latency (s)'):.3f}",
                f"{s.get('Input token throughput (tok/s)', 0):.2f}",
                f"{s.get('Average input tokens per request', 0):.1f}",
                f'{r.success_rate:.1f}%',
            ])
    else:
        columns = [
            'Conc.',
            'Rate',
            'RPS',
            'Avg Lat.(s)',
            'P99 Lat.(s)',
            'Avg TTFT(s)',
            'P99 TTFT(s)',
            'Avg TPOT(s)',
            'P99 TPOT(s)',
            'Gen. tok/s',
            'Success Rate',
        ]
        rows = []
        for r in runs:
            s = r.summary
            rate = s.get('Request rate (req/s)', -1)
            rows.append([
                str(int(s.get('Number of concurrency', 0))),
                'INF' if rate == -1 else f'{rate:.2f}',
                f"{s.get('Request throughput (req/s)', 0):.4f}",
                f"{s.get('Average latency (s)', 0):.3f}",
                f"{r.get_p99('Latency (s)'):.3f}",
                f"{s.get('Average time to first token (s)', 0):.3f}",
                f"{r.get_p99('TTFT (s)'):.3f}",
                f"{s.get('Average time per output token (s)', 0):.3f}",
                f"{r.get_p99('TPOT (s)'):.3f}",
                f"{s.get('Output token throughput (tok/s)', 0):.2f}",
                f'{r.success_rate:.1f}%',
            ])

    return columns, rows


def _build_best_config(runs: list) -> OrderedDict:
    """Return best-RPS and lowest-latency configurations."""
    if not runs:
        return OrderedDict()

    best: OrderedDict = OrderedDict()

    best_rps = max(runs, key=lambda r: r.summary.get('Request throughput (req/s)', 0))
    best['Highest RPS'] = (
        f'Concurrency {best_rps.parallel} '
        f"({best_rps.summary.get('Request throughput (req/s)', 0):.4f} req/s)"
    )

    best_lat = min(runs, key=lambda r: r.summary.get('Average latency (s)', float('inf')))
    best['Lowest Latency'] = (
        f'Concurrency {best_lat.parallel} '
        f"({best_lat.summary.get('Average latency (s)', 0):.3f} s)"
    )

    return best


def _build_recommendations(runs: list) -> List[str]:
    """Generate human-readable performance recommendations."""
    if not runs:
        return []

    recs: List[str] = []
    sorted_runs = sorted(runs, key=lambda r: r.parallel)
    rps_values = [r.summary.get('Request throughput (req/s)', 0) for r in sorted_runs]

    if len(rps_values) >= 2:
        best_idx = rps_values.index(max(rps_values))
        if best_idx == len(rps_values) - 1:
            recs.append(
                'The system has not reached its performance bottleneck. '
                'Consider testing with higher concurrency levels.'
            )
        elif best_idx == 0:
            recs.append('Consider lowering concurrency; load may be too high for the system.')
        else:
            recs.append(f'Optimal concurrency appears to be around {sorted_runs[best_idx].parallel}.')

    last = sorted_runs[-1]
    if last.success_rate < 95:
        recs.append(
            f'Success rate at highest concurrency ({last.parallel}) is {last.success_rate:.1f}%. '
            'Check system resources or reduce concurrency.'
        )

    if len(sorted_runs) >= 2:
        first_lat = sorted_runs[0].summary.get('Average latency (s)', 0)
        last_lat = sorted_runs[-1].summary.get('Average latency (s)', 0)
        if first_lat > 0 and last_lat / first_lat > 3:
            recs.append(
                f'Latency grew {last_lat / first_lat:.1f}× from lowest to highest concurrency. '
                'The system may be under significant load.'
            )

    return recs


# ---------------------------------------------------------------------------
# Per-run section builder
# ---------------------------------------------------------------------------


def _build_run_section(run, is_embedding_flag: bool) -> Dict[str, Any]:
    """Build the template-ready dict for one per-run accordion card."""
    from . import perf_charts as charts

    pct = run.percentiles
    return {
        'name': run.name,
        'total_requests': run.summary.get('Total requests', 0),
        # benchmark_summary.json displayed as a stat grid
        'summary_items': run.summary_items(is_embedding_flag),
        # percentile table
        'percentile_columns': list(pct[0].keys()) if pct else [],
        'percentile_rows': [list(p.values()) for p in pct],
        # charts
        'percentile_chart': charts.build_percentile_chart(run, is_embedding_flag),
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
    is_emb = _is_embedding(api_type)
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
    summary_columns, summary_rows = _build_summary_table(runs, is_emb)

    # ── Template rendering ──────────────────────────────────────────────────
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=False)
    template = env.get_template('perf_report.html.j2')

    html_content = template.render(
        model=first_args.get('model', first_args.get('model_id', 'N/A')),
        api_type=api_type,
        dataset=first_args.get('dataset', 'N/A'),
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        evalscope_version=_evalscope_version,
        basic_info=_build_basic_info(first_args, runs, is_emb),
        summary_columns=summary_columns,
        summary_rows=summary_rows,
        best_config=_build_best_config(runs),
        recommendations=_build_recommendations(runs),
        latency_tabs=latency_tabs,
        throughput_tabs=throughput_tabs,
        run_sections=[_build_run_section(r, is_emb) for r in runs],
    )

    out_path = os.path.join(output_dir, output_html_name)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(html_content)

    logger.info(f'Performance HTML report generated: {out_path}')
    return out_path
