"""Public helpers for building perf-run summary view-models.

These builders operate purely on :class:`~evalscope.perf.utils.perf_models.RunData`
aggregates (summary + percentiles) and never touch the per-request DB rows, so
they are cheap to call from both the HTML report generator
(:mod:`evalscope.perf.utils.report.generate_report`) and the web-service archive
endpoints. Promoting them to this public module lets both render identical
Basic-Info / summary-table / recommendations without importing private
underscored helpers.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List


def is_embedding(api_type: str) -> bool:
    """Return True when *api_type* indicates an embedding / rerank model."""
    lower = (api_type or '').lower()
    return 'embedding' in lower or 'rerank' in lower or 'embed' in lower


def build_basic_info(
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


def build_summary_table(runs: list, is_embedding_flag: bool):
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


def build_best_config(runs: list) -> OrderedDict:
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


def build_recommendations(runs: list) -> List[str]:
    """Generate human-readable performance recommendations."""
    if not runs:
        return []

    recs: List[str] = []
    sorted_runs = sorted(runs, key=lambda r: r.sort_key)
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


def build_summary_items(
    summary,
    is_embedding_flag: bool,
) -> List[Dict[str, str]]:
    """Format *summary* fields into ``[{'key': ..., 'value': ...}]`` for the stat grid.

    Owns display-layer logic: unit conversion (ms), INF substitution, and format
    strings.
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

    if is_embedding_flag:
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
