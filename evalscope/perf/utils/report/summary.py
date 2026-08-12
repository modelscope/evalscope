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
from typing import Any, Dict, List

from evalscope.metrics.semantics import format_perf_value, resolve_perf_semantics
from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics

_CONFIG_COLUMNS = [
    ('concurrency', 'Conc.', None),
    ('request_rate', 'Rate', None),
]
_COMMON_METRIC_COLUMNS = [
    ('request_throughput', 'RPS', Metrics.REQUEST_THROUGHPUT),
    ('avg_latency', 'Avg Lat.(s)', Metrics.AVERAGE_LATENCY),
    ('p99_latency', 'P99 Lat.(s)', PercentileMetrics.LATENCY),
]
_EMBEDDING_METRIC_COLUMNS = [
    ('input_token_throughput', 'Avg Inp.TPS', Metrics.INPUT_TOKEN_THROUGHPUT),
    ('avg_input_tokens', 'Avg Inp.Tok', Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST),
]
_GENERATION_METRIC_COLUMNS = [
    ('avg_ttft', 'Avg TTFT(ms)', Metrics.AVERAGE_TIME_TO_FIRST_TOKEN),
    ('p99_ttft', 'P99 TTFT(ms)', PercentileMetrics.TTFT),
    ('avg_tpot', 'Avg TPOT(ms)', Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN),
    ('p99_tpot', 'P99 TPOT(ms)', PercentileMetrics.TPOT),
    ('output_token_throughput', 'Gen. tok/s', Metrics.OUTPUT_TOKEN_THROUGHPUT),
]
_SUCCESS_COLUMN = ('success_rate', 'Success Rate', 'success_rate')


def _cell(field_key: str, value: float, include_unit: bool = False) -> str:
    """Format one perf table cell through the semantics registry."""
    return format_perf_value(value, field_key, include_unit=include_unit)


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
    info['Total Test Time'] = _cell(Metrics.TIME_TAKEN_FOR_TESTS, total_time, include_unit=True)

    if is_embedding_flag:
        total_input = sum(r.summary.avg_input_tokens * r.summary.succeed_requests for r in runs)
        info['Total Input Tokens'] = f'{total_input:,.0f}'
    else:
        total_output = sum(r.summary.avg_output_tokens * r.summary.succeed_requests for r in runs)
        info['Total Output Tokens'] = f'{total_output:,.0f}'

    return info


def build_summary_table(runs: list, is_embedding_flag: bool):
    """Build the cross-run summary table. Returns *(columns, rows)*."""
    specs = _CONFIG_COLUMNS + _COMMON_METRIC_COLUMNS
    if is_embedding_flag:
        specs += _EMBEDDING_METRIC_COLUMNS + [_SUCCESS_COLUMN]
        rows = []
        for r in runs:
            s = r.summary
            rate = s.request_rate
            rows.append([
                'INF' if s.concurrency == -1 else str(s.concurrency),
                'INF' if rate == -1 else _cell(Metrics.REQUEST_RATE, rate),
                _cell(Metrics.REQUEST_THROUGHPUT, s.request_throughput),
                _cell(Metrics.AVERAGE_LATENCY, s.avg_latency),
                _cell(PercentileMetrics.LATENCY, r.get_p99('latency')),
                _cell(Metrics.INPUT_TOKEN_THROUGHPUT, s.input_token_throughput),
                _cell(Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST, s.avg_input_tokens),
                _cell('success_rate', r.success_rate, include_unit=True),
            ])
    else:
        specs += _GENERATION_METRIC_COLUMNS + [_SUCCESS_COLUMN]
        rows = []
        for r in runs:
            s = r.summary
            rate = s.request_rate
            rows.append([
                'INF' if s.concurrency == -1 else str(s.concurrency),
                'INF' if rate == -1 else _cell(Metrics.REQUEST_RATE, rate),
                _cell(Metrics.REQUEST_THROUGHPUT, s.request_throughput),
                _cell(Metrics.AVERAGE_LATENCY, s.avg_latency),
                _cell(PercentileMetrics.LATENCY, r.get_p99('latency')),
                _cell(Metrics.AVERAGE_TIME_TO_FIRST_TOKEN, s.avg_ttft),
                _cell(PercentileMetrics.TTFT, r.get_p99('ttft')),
                _cell(Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN, s.avg_tpot),
                _cell(PercentileMetrics.TPOT, r.get_p99('tpot')),
                _cell(Metrics.OUTPUT_TOKEN_THROUGHPUT, s.output_token_throughput),
                _cell('success_rate', r.success_rate, include_unit=True),
            ])

    semantics = resolve_perf_semantics(field_key for _, _, field_key in specs if field_key is not None)
    columns: List[Dict[str, Any]] = [
        {
            'key': key,
            'label': label,
            'semantics': semantics.get(field_key) if field_key is not None else None,
        } for key, label, field_key in specs
    ]
    return columns, rows


def build_best_config(runs: list) -> OrderedDict:
    """Return best-RPS and lowest-latency configurations."""
    if not runs:
        return OrderedDict()

    best: OrderedDict = OrderedDict()

    best_rps = max(runs, key=lambda r: r.summary.request_throughput)
    best['Highest RPS'] = (
        f'{best_rps.name} '
        f'({_cell(Metrics.REQUEST_THROUGHPUT, best_rps.summary.request_throughput, include_unit=True)})'
    )

    best_lat = min(runs, key=lambda r: r.summary.avg_latency if r.summary.avg_latency >= 0 else float('inf'))
    best['Lowest Latency'] = (
        f'{best_lat.name} '
        f'({_cell(Metrics.AVERAGE_LATENCY, best_lat.summary.avg_latency, include_unit=True)})'
    )

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
            f'Success rate at highest load ({last.name}) is '
            f'{_cell("success_rate", last.success_rate, include_unit=True)}. '
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
    rate_str = 'INF' if rate_raw == -1 else _cell(Metrics.REQUEST_RATE, rate_raw)
    concurrency_str = 'INF' if s.concurrency == -1 else str(s.concurrency)

    base = [
        ('Total Requests', _cell(Metrics.TOTAL_REQUESTS, s.total_requests)),
        ('Succeed Requests', _cell(Metrics.SUCCEED_REQUESTS, s.succeed_requests)),
        ('Failed Requests', _cell(Metrics.FAILED_REQUESTS, s.failed_requests)),
        ('Concurrency', concurrency_str),
        ('Time Taken (s)', _cell(Metrics.TIME_TAKEN_FOR_TESTS, s.time_taken)),
        ('Request Rate (req/s)', rate_str),
        ('Request Throughput (req/s)', _cell(Metrics.REQUEST_THROUGHPUT, s.request_throughput)),
        ('Avg Latency (s)', _cell(Metrics.AVERAGE_LATENCY, s.avg_latency)),
    ]

    if is_embedding_flag:
        extra = [
            ('Input Tok Throughput (tok/s)', _cell(Metrics.INPUT_TOKEN_THROUGHPUT, s.input_token_throughput)),
            ('Avg Input Tokens', _cell(Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST, s.avg_input_tokens)),
        ]
    else:
        extra = [
            ('Output Tok Throughput (tok/s)', _cell(Metrics.OUTPUT_TOKEN_THROUGHPUT, s.output_token_throughput)),
            ('Total Tok Throughput (tok/s)', _cell(Metrics.TOTAL_TOKEN_THROUGHPUT, s.total_token_throughput)),
            ('Avg TTFT (ms)', _cell(Metrics.AVERAGE_TIME_TO_FIRST_TOKEN, s.avg_ttft)),
            ('Avg TPOT (ms)', _cell(Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN, s.avg_tpot)),
            ('Avg ITL (ms)', _cell(Metrics.AVERAGE_INTER_TOKEN_LATENCY, s.avg_itl)),
            ('Avg Input Tokens', _cell(Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST, s.avg_input_tokens)),
            ('Avg Output Tokens', _cell(Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST, s.avg_output_tokens)),
        ]

    return [{'key': k, 'value': v} for k, v in base + extra]
