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

from evalscope.constants import DEFAULT_LANGUAGE
from evalscope.utils.io_utils import current_time
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

    total_req = sum(r.summary.total_requests for r in runs)
    succeed = sum(r.summary.succeed_requests for r in runs)
    total_time = sum(r.summary.time_taken for r in runs)

    info['Total Requests'] = f'{total_req:,}'
    info['Succeed Requests'] = f'{succeed:,}'
    info['Total Test Time'] = f'{total_time:.2f} s'

    if is_embedding_flag:
        total_input = sum(r.summary.avg_input_tokens * r.summary.succeed_requests for r in runs)
        info['Total Input Tokens'] = f'{total_input:,.0f}'
    else:
        total_output = sum(r.summary.avg_output_tokens * r.summary.succeed_requests for r in runs)
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
            rate = s.request_rate
            rows.append([
                'INF' if s.concurrency == -1 else str(s.concurrency),
                'INF' if rate == -1 else f'{rate:.2f}',
                f'{s.request_throughput:.4f}',
                f'{s.avg_latency:.3f}',
                f'{r.get_p99("latency"):.3f}',
                f'{s.input_token_throughput:.2f}',
                f'{s.avg_input_tokens:.1f}',
                f'{r.success_rate:.1f}%',
            ])
    else:
        columns = [
            'Conc.',
            'Rate',
            'RPS',
            'Avg Lat.(s)',
            'P99 Lat.(s)',
            'Avg TTFT(ms)',
            'P99 TTFT(ms)',
            'Avg TPOT(ms)',
            'P99 TPOT(ms)',
            'Gen. tok/s',
            'Success Rate',
        ]
        rows = []
        for r in runs:
            s = r.summary
            rate = s.request_rate
            rows.append([
                'INF' if s.concurrency == -1 else str(s.concurrency),
                'INF' if rate == -1 else f'{rate:.2f}',
                f'{s.request_throughput:.4f}',
                f'{s.avg_latency:.3f}',
                f'{r.get_p99("latency"):.3f}',
                f'{s.avg_ttft:.2f}',
                f'{r.get_p99("ttft"):.2f}',
                f'{s.avg_tpot:.2f}',
                f'{r.get_p99("tpot"):.2f}',
                f'{s.output_token_throughput:.2f}',
                f'{r.success_rate:.1f}%',
            ])

    return columns, rows


def _build_best_config(runs: list) -> OrderedDict:
    """Return best-RPS and lowest-latency configurations."""
    if not runs:
        return OrderedDict()

    best: OrderedDict = OrderedDict()

    best_rps = max(runs, key=lambda r: r.summary.request_throughput)
    best['Highest RPS'] = (f'{best_rps.name} '
                           f'({best_rps.summary.request_throughput:.4f} req/s)')

    best_lat = min(runs, key=lambda r: r.summary.avg_latency if r.summary.avg_latency >= 0 else float('inf'))
    best['Lowest Latency'] = (f'{best_lat.name} '
                              f'({best_lat.summary.avg_latency:.3f} s)')

    return best


def _build_recommendations(runs: list) -> List[str]:
    """Generate human-readable performance recommendations."""
    if not runs:
        return []

    recs: List[str] = []
    sorted_runs = sorted(runs, key=lambda r: r.parallel)
    rps_values = [r.summary.request_throughput for r in sorted_runs]

    if len(rps_values) >= 2:
        best_idx = rps_values.index(max(rps_values))
        if best_idx == len(rps_values) - 1:
            recs.append(
                'The system has not reached its performance bottleneck. '
                'Consider testing with higher load levels.'
            )
        elif best_idx == 0:
            recs.append('Consider lowering the load; it may be too high for the system.')
        else:
            recs.append(f'Optimal configuration appears to be around {sorted_runs[best_idx].name}.')

    last = sorted_runs[-1]
    if last.success_rate < 95:
        recs.append(
            f'Success rate at highest load ({last.name}) is {last.success_rate:.1f}%. '
            'Check system resources or reduce the load.'
        )

    if len(sorted_runs) >= 2:
        first_lat = sorted_runs[0].summary.avg_latency
        last_lat = sorted_runs[-1].summary.avg_latency
        if first_lat > 0 and last_lat / first_lat > 3:
            recs.append(
                f'Latency grew {last_lat / first_lat:.1f}\u00d7 from lowest to highest load. '
                'The system may be under significant stress.'
            )

    return recs


# ---------------------------------------------------------------------------
# Per-run section builder
# ---------------------------------------------------------------------------


def _build_summary_items(
    summary,
    is_embedding: bool,
) -> List[Dict[str, str]]:
    """Format *summary* fields into ``[{'key': ..., 'value': ...}]`` for the stat grid.

    Kept here (not in perf_models) because it owns display-layer logic:
    unit conversion (ms), INF substitution, and format strings.
    """
    s = summary
    rate_raw = s.request_rate
    rate_str = 'INF' if rate_raw == -1 else f'{rate_raw:.3f}'
    concurrency_str = 'INF' if s.concurrency == -1 else str(s.concurrency)

    base = [
        ('Total Requests', str(s.total_requests)),
        ('Succeed Requests', str(s.succeed_requests)),
        ('Failed Requests', str(s.failed_requests)),
        ('Concurrency', concurrency_str),
        ('Time Taken (s)', f'{s.time_taken:.3f}'),
        ('Request Rate (req/s)', rate_str),
        ('Request Throughput (req/s)', f'{s.request_throughput:.4f}'),
        ('Avg Latency (s)', f'{s.avg_latency:.4f}'),
    ]

    if is_embedding:
        extra = [
            ('Input Tok Throughput (tok/s)', f'{s.input_token_throughput:.2f}'),
            ('Avg Input Tokens', f'{s.avg_input_tokens:.1f}'),
        ]
    else:
        extra = [
            ('Output Tok Throughput (tok/s)', f'{s.output_token_throughput:.2f}'),
            ('Total Tok Throughput (tok/s)', f'{s.total_token_throughput:.2f}'),
            ('Avg TTFT (ms)', f'{s.avg_ttft:.2f}'),
            ('Avg TPOT (ms)', f'{s.avg_tpot:.2f}'),
            ('Avg ITL (ms)', f'{s.avg_itl:.2f}'),
            ('Avg Input Tokens', f'{s.avg_input_tokens:.1f}'),
            ('Avg Output Tokens', f'{s.avg_output_tokens:.1f}'),
        ]

    return [{'key': k, 'value': v} for k, v in base + extra]


def _build_run_section(run, is_embedding_flag: bool) -> Dict[str, Any]:
    """Build the template-ready dict for one per-run accordion card."""
    from . import perf_charts as charts

    pct = run.percentiles.to_list()
    pct_charts = charts.build_percentile_chart(run, is_embedding_flag)
    return {
        'name': run.name,
        'total_requests': run.summary.total_requests,
        # benchmark_summary.json displayed as a stat grid
        'summary_items': _build_summary_items(run.summary, is_embedding_flag),
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
        generated_at=current_time().strftime('%Y-%m-%d %H:%M:%S'),
        evalscope_version=_evalscope_version,
        basic_info=_build_basic_info(first_args, runs, is_emb),
        summary_columns=summary_columns,
        summary_rows=summary_rows,
        best_config=_build_best_config(runs),
        recommendations=_build_recommendations(runs),
        latency_tabs=latency_tabs,
        throughput_tabs=throughput_tabs,
        run_sections=[_build_run_section(r, is_emb) for r in runs],
        default_lang=DEFAULT_LANGUAGE,
    )

    out_path = os.path.join(output_dir, output_html_name)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(html_content)

    return out_path
