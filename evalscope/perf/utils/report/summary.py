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
from typing import Any, Dict, List, Optional

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


def _summary_specs(is_embedding_flag: bool) -> list:
    """Return ordered summary-column specifications for the API type."""
    specs = _CONFIG_COLUMNS + _COMMON_METRIC_COLUMNS
    if is_embedding_flag:
        specs += _EMBEDDING_METRIC_COLUMNS + [_SUCCESS_COLUMN]
    else:
        specs += _GENERATION_METRIC_COLUMNS + [_SUCCESS_COLUMN]
    return specs


def _summary_values(run: Any, is_embedding_flag: bool) -> Dict[str, float]:
    """Return unformatted summary values keyed by stable API field names."""
    summary = run.summary
    values = {
        'concurrency': summary.concurrency,
        'request_rate': summary.request_rate,
        'request_throughput': summary.request_throughput,
        'avg_latency': summary.avg_latency,
        'p99_latency': run.get_p99('latency'),
    }
    if is_embedding_flag:
        values.update(
            {
                'input_token_throughput': summary.input_token_throughput,
                'avg_input_tokens': summary.avg_input_tokens,
            }
        )
    else:
        values.update(
            {
                'avg_ttft': summary.avg_ttft,
                'p99_ttft': run.get_p99('ttft'),
                'avg_tpot': summary.avg_tpot,
                'p99_tpot': run.get_p99('tpot'),
                'output_token_throughput': summary.output_token_throughput,
            }
        )
    values['success_rate'] = run.success_rate
    return values


def _summary_sample_counts(run: Any, request_counts: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Return the effective observation count for every summary metric."""
    summary = run.summary
    request_counts = request_counts or {}
    total = request_counts.get('total', summary.total_requests)
    successful = request_counts.get('successful', summary.succeed_requests)
    stream_successful = request_counts.get('stream_successful')
    if stream_successful is None:
        stream_successful = min(successful, summary.stream_requests) if summary.stream_requests > 0 else successful
    generation_successful = stream_successful if stream_successful > 0 else successful

    return {
        'request_throughput': successful,
        'avg_latency': successful,
        'p99_latency': successful,
        'input_token_throughput': successful,
        'avg_input_tokens': successful,
        'avg_ttft': generation_successful,
        'p99_ttft': generation_successful,
        'avg_tpot': generation_successful,
        'p99_tpot': generation_successful,
        'output_token_throughput': successful,
        'success_rate': total,
    }


def build_summary_table(
    runs: list,
    is_embedding_flag: bool,
    request_counts: Optional[List[Optional[Dict[str, int]]]] = None,
) -> tuple:
    """Build a structured, unformatted cross-run summary table."""
    specs = _summary_specs(is_embedding_flag)

    semantics = resolve_perf_semantics(field_key for _, _, field_key in specs if field_key is not None)
    columns: List[Dict[str, Any]] = [
        {
            'key': key,
            'label': label,
            'semantics': semantics.get(field_key) if field_key is not None else None,
        }
        for key, label, field_key in specs
    ]
    rows = [
        {
            'values': _summary_values(run, is_embedding_flag),
            'sample_counts': _summary_sample_counts(run, request_counts[index] if request_counts else None),
        }
        for index, run in enumerate(runs)
    ]
    return columns, rows


def format_summary_rows(columns: list, rows: list) -> List[List[str]]:
    """Format structured summary rows for the standalone HTML report."""
    all_specs = (
        _CONFIG_COLUMNS
        + _COMMON_METRIC_COLUMNS
        + _EMBEDDING_METRIC_COLUMNS
        + _GENERATION_METRIC_COLUMNS
        + [_SUCCESS_COLUMN]
    )
    field_keys = {key: field_key for key, _, field_key in all_specs}
    formatted_rows: List[List[str]] = []
    for row in rows:
        formatted_row = []
        for column in columns:
            key = column['key']
            value = row['values'][key]
            if key in ('concurrency', 'request_rate') and value == -1:
                formatted_row.append('INF')
            elif key == 'concurrency':
                formatted_row.append(str(int(value)))
            else:
                formatted_row.append(_cell(field_keys[key], value, include_unit=key == 'success_rate'))
        formatted_rows.append(formatted_row)
    return formatted_rows


def build_best_config(runs: list) -> OrderedDict:
    """Return best-RPS and lowest-latency configurations."""
    if not runs:
        return OrderedDict()

    best: OrderedDict = OrderedDict()

    best_rps = max(runs, key=lambda r: r.summary.request_throughput)
    best['Highest RPS'] = (
        f'{best_rps.name} ({_cell(Metrics.REQUEST_THROUGHPUT, best_rps.summary.request_throughput, include_unit=True)})'
    )

    best_lat = min(runs, key=lambda r: r.summary.avg_latency if r.summary.avg_latency >= 0 else float('inf'))
    best['Lowest Latency'] = (
        f'{best_lat.name} ({_cell(Metrics.AVERAGE_LATENCY, best_lat.summary.avg_latency, include_unit=True)})'
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
                'The system has not reached its performance bottleneck. Consider testing with higher load levels.'
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
