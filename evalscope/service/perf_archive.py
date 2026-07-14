"""Historical perf-run archive: discovery, view-models, charts, request paging.

This module holds all the non-HTTP logic behind the ``/api/v1/perf/*`` archive
endpoints so the Flask blueprint stays thin (route parsing + JSON/error
translation only). Functions raise :class:`PerfArchiveError` with an HTTP status
so the blueprint can map failures to responses uniformly.

Loading is metadata-only by default (summary + percentiles); per-request DB rows
are read lazily and page-by-page (see :func:`query_request_page`) so list/detail
endpoints never pull entire historical request tables into memory.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from types import SimpleNamespace
from typing import List, Optional

from evalscope.constants import PLOTLY_CDN_URL
from evalscope.perf.utils.report.summary import (
    build_basic_info,
    build_best_config,
    build_recommendations,
    build_summary_table,
    is_embedding,
)
from evalscope.utils.logger import get_logger

logger = get_logger()

# Recognized perf run-directory markers (mirror perf_data.py).
_PARALLEL_RE = re.compile(r'^parallel_(\d+)_number_(\d+)$')
_RATE_RE = re.compile(r'^rate_([\d.]+)_number_(\d+)$')
# Bound recursive scan depth so large output trees stay cheap to walk.
MAX_SCAN_DEPTH = 3

# Sweep chart_type -> (builder attr, needs is_embedding flag). One point per run.
SWEEP_CHARTS = {
    'latency': ('build_latency_chart', False),
    'ttft': ('build_ttft_chart', False),
    'tpot': ('build_tpot_chart', False),
    'rps': ('build_rps_chart', False),
    'throughput': ('build_throughput_chart', True),
    'success': ('build_success_chart', False),
}

# Per-run request-detail chart_type -> tab label from build_request_detail_tabs.
REQUEST_CHARTS = {
    'req_latency': 'Latency',
    'req_ttft_tpot': 'TTFT / TPOT / ITL',
    'req_tokens': 'Tokens',
    'req_success': 'Success',
}

# Per-run percentile chart_type -> index into build_percentile_chart() tuple.
PERCENTILE_CHARTS = {'percentile_latency': 0, 'percentile_token': 1}


class PerfArchiveError(Exception):
    """Domain error carrying an HTTP status for the blueprint to translate."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


# ------------------------------------------------------------------
# Discovery / path resolution
# ------------------------------------------------------------------


def is_run_dir(entry_path: str) -> bool:
    """Return True when *entry_path* is a perf-run directory.

    A perf-run directory either contains ``perf_report.html`` directly or has at
    least one ``parallel_*_number_*`` / ``rate_*_number_*`` sub-directory that
    holds a ``benchmark_summary.json`` file.
    """
    if os.path.isfile(os.path.join(entry_path, 'perf_report.html')):
        return True
    try:
        for child in os.listdir(entry_path):
            if not (_PARALLEL_RE.match(child) or _RATE_RE.match(child)):
                continue
            if os.path.isfile(os.path.join(entry_path, child, 'benchmark_summary.json')):
                return True
    except OSError:
        return False
    return False


def scan_perf_runs(root: str) -> List[str]:
    """Recursively discover perf-run directories under *root* (bounded depth).

    Returns a sorted (desc) list of paths relative to *root*. Covers both the
    CLI layout (``<ts>/<model>/``) and the service layout (``<task_id>/perf/``)
    because both share the same marker files.
    """
    if not root or not os.path.isdir(root):
        return []

    root = os.path.abspath(root)
    root_real = os.path.realpath(root)
    found: List[str] = []

    def _walk(current: str, depth: int) -> None:
        if depth > MAX_SCAN_DEPTH:
            return
        try:
            entries = sorted(os.listdir(current))
        except OSError:
            return
        for name in entries:
            entry_path = os.path.join(current, name)
            if not os.path.isdir(entry_path):
                continue
            # Reject entries whose realpath escapes the outputs root (e.g. via a
            # symlink pointing outside the tree) to avoid reading foreign files.
            if not os.path.realpath(entry_path).startswith(root_real + os.sep):
                continue
            if is_run_dir(entry_path):
                found.append(os.path.relpath(entry_path, root))
                # Do not descend further: its children are individual runs.
                continue
            _walk(entry_path, depth + 1)

    _walk(root, 0)
    return sorted(found, reverse=True)


def extract_timestamp(rel_path: str, abs_path: str) -> str:
    """Parse a timestamp from a path segment (YYYYMMDD_HHMMSS) or fall back to mtime."""
    for seg in rel_path.replace('\\', '/').split('/'):
        for fmt in ('%Y%m%d_%H%M%S', '%Y%m%d'):
            try:
                return datetime.strptime(seg, fmt).isoformat()
            except ValueError:
                continue
    try:
        return datetime.fromtimestamp(os.path.getmtime(abs_path)).isoformat()
    except OSError:
        return ''


def resolve_run_dir(root: str, rel_path: str) -> Optional[str]:
    """Resolve *rel_path* against *root*, rejecting path traversal.

    Returns the absolute directory path, or ``None`` when the resolved path
    escapes *root* or is not a directory.
    """
    root_real = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root_real, rel_path))
    if target != root_real and not target.startswith(root_real + os.sep):
        return None
    if not os.path.isdir(target):
        return None
    return target


def _load_runs(run_dir: str, *, with_requests: bool = False):
    """Load runs from *run_dir* via RunLoader (lazy import to avoid cycles).

    Defaults to metadata-only loading; per-request DB rows are skipped unless
    *with_requests* is set.
    """
    from evalscope.perf.utils.report.perf_data import RunLoader
    return RunLoader.load_all(run_dir, with_requests=with_requests)


def _find_run(runs, run_name: str):
    """Return the RunData whose dir_name matches *run_name* (regex-validated), or None."""
    if not (_PARALLEL_RE.match(run_name) or _RATE_RE.match(run_name)):
        return None
    for r in runs:
        if r.dir_name == run_name:
            return r
    return None


# ------------------------------------------------------------------
# View-models
# ------------------------------------------------------------------


def build_run_summary(rel_path: str, abs_path: str) -> Optional[dict]:
    """Build lightweight list-item metadata for one perf-run directory."""
    runs = _load_runs(abs_path, with_requests=False)
    has_html = os.path.isfile(os.path.join(abs_path, 'perf_report.html'))
    if not runs:
        if not has_html:
            return None
        return {
            'path': rel_path,
            'model': os.path.basename(rel_path.rstrip('/')),
            'api_type': '',
            'dataset': '',
            'num_runs': 0,
            'total_requests': 0,
            'success_rate': 0.0,
            'best_rps': 0.0,
            'best_latency': 0.0,
            'is_embedding': False,
            'has_html': has_html,
            'timestamp': extract_timestamp(rel_path, abs_path),
        }

    first_args = runs[0].args or {}
    api_type = first_args.get('api', '')
    total_requests = sum(r.summary.total_requests for r in runs)
    total_succeed = sum(r.summary.succeed_requests for r in runs)
    success_rate = round(total_succeed / total_requests * 100, 1) if total_requests else 0.0
    best_rps = max(r.summary.request_throughput for r in runs)
    valid_lat = [r.summary.avg_latency for r in runs if r.summary.avg_latency >= 0]
    best_latency = min(valid_lat) if valid_lat else 0.0

    return {
        'path': rel_path,
        'model': first_args.get('model', first_args.get('model_id', 'N/A')),
        'api_type': api_type,
        'dataset': first_args.get('dataset', ''),
        'num_runs': len(runs),
        'total_requests': total_requests,
        'success_rate': success_rate,
        'best_rps': round(best_rps, 4),
        'best_latency': round(best_latency, 4),
        'is_embedding': is_embedding(api_type),
        'has_html': has_html,
        'timestamp': extract_timestamp(rel_path, abs_path),
    }


def list_run_summaries(root: str) -> List[dict]:
    """Scan *root* and return list-item metadata for every perf run (sorted desc)."""
    root_abs = os.path.abspath(root)
    runs: List[dict] = []
    for rel_path in scan_perf_runs(root_abs):
        try:
            meta = build_run_summary(rel_path, os.path.join(root_abs, rel_path))
        except Exception as e:
            logger.warning(f'Skipping unreadable perf run {rel_path}: {e}')
            continue
        if meta is not None:
            runs.append(meta)
    runs.sort(key=lambda x: x['timestamp'], reverse=True)
    return runs


def build_run_detail(root: str, rel_path: str) -> dict:
    """Build native-render metadata for a single perf-run directory."""
    run_dir = resolve_run_dir(root, rel_path)
    if run_dir is None:
        raise PerfArchiveError('Invalid path', 400)

    runs = _load_runs(run_dir, with_requests=False)
    if not runs:
        raise PerfArchiveError(f'No perf runs found under: {rel_path}', 404)

    first_args = runs[0].args or {}
    api_type = first_args.get('api', '')
    is_emb = is_embedding(api_type)
    summary_columns, summary_rows = build_summary_table(runs, is_emb)

    return {
        'path': rel_path,
        'model': first_args.get('model', first_args.get('model_id', 'N/A')),
        'api_type': api_type,
        'dataset': first_args.get('dataset', 'N/A'),
        'generated_at': extract_timestamp(rel_path, run_dir),
        'basic_info': dict(build_basic_info(first_args, runs, is_emb)),
        'summary_columns': summary_columns,
        'summary_rows': summary_rows,
        'best_config': dict(build_best_config(runs)),
        'recommendations': build_recommendations(runs),
        'num_runs': len(runs),
        'is_embedding': is_emb,
        'has_html': os.path.isfile(os.path.join(run_dir, 'perf_report.html')),
    }


def list_run_items(root: str, rel_path: str) -> List[dict]:
    """List individual runs (parallel_*/rate_*) within a perf-run directory.

    Request counts come from a cheap ``COUNT(*)`` per run rather than loading
    every request row.
    """
    from evalscope.perf.utils.report.perf_data import RunLoader

    run_dir = resolve_run_dir(root, rel_path)
    if run_dir is None:
        raise PerfArchiveError('Invalid path', 400)

    runs = _load_runs(run_dir, with_requests=False)
    items: List[dict] = []
    for r in runs:
        pct = r.percentiles.to_list()
        num_requests = RunLoader.count_requests(os.path.join(run_dir, r.dir_name))
        items.append({
            'dir_name': r.dir_name,
            'name': r.name,
            'parallel': r.parallel,
            'number': r.number,
            'rate': r.rate,
            'total_requests': r.summary.total_requests,
            'succeed_requests': r.summary.succeed_requests,
            'success_rate': r.success_rate,
            'num_requests': num_requests,
            'has_requests': num_requests > 0,
            'percentile_columns': list(pct[0].keys()) if pct else [],
            'percentile_rows': [list(p.values()) for p in pct],
        })
    return items


def query_request_page(
    root: str,
    rel_path: str,
    run_name: str,
    status: Optional[str],
    page: int,
    page_size: int,
) -> dict:
    """Return a paginated page of per-request records for one run (DB-backed)."""
    from evalscope.perf.utils.report.perf_data import RunLoader

    run_dir = resolve_run_dir(root, rel_path)
    if run_dir is None:
        raise PerfArchiveError('Invalid path', 400)
    if not (_PARALLEL_RE.match(run_name) or _RATE_RE.match(run_name)):
        raise PerfArchiveError(f'Run not found: {run_name}', 404)
    sub_dir = os.path.join(run_dir, run_name)
    if not os.path.isdir(sub_dir):
        raise PerfArchiveError(f'Run not found: {run_name}', 404)

    status = status if status in ('success', 'failed') else None
    page = max(1, page)
    page_size = max(1, min(500, page_size))
    offset = (page - 1) * page_size

    records, total = RunLoader.query_requests(sub_dir, status=status, offset=offset, limit=page_size)
    # Unfiltered row count drives has_db; avoid an extra query when unfiltered.
    db_total = RunLoader.count_requests(sub_dir) if status else total

    rows = []
    for i, r in enumerate(records):
        rows.append({
            '#': offset + i + 1,
            'Latency(s)': round(r.latency, 3),
            'TTFT(ms)': round(r.first_chunk_latency, 1) if r.first_chunk_latency is not None else None,
            'TPOT(ms)': round(r.time_per_output_token, 1) if r.time_per_output_token is not None else None,
            'Prompt': r.prompt_tokens,
            'Completion': r.completion_tokens,
            'Success': 'OK' if r.success else 'FAIL',
        })

    return {
        'columns': ['#', 'Latency(s)', 'TTFT(ms)', 'TPOT(ms)', 'Prompt', 'Completion', 'Success'],
        'rows': rows,
        'total': total,
        'page': page,
        'page_size': page_size,
        'has_db': db_total > 0,
    }


# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------


def wrap_chart_html(div: str) -> str:
    """Wrap a Plotly <div> into a standalone HTML page that fills the iframe.

    The Plotly graph div is emitted with ``height:100%``; without giving the
    html/body and the wrapping div a definite height it collapses to Plotly's
    default size and overflows the fixed-height iframe (scrollbars + clipping).
    Forcing the full chain to 100% height and hiding overflow makes the chart
    fill the iframe cleanly.
    """
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<script src="{PLOTLY_CDN_URL}" charset="utf-8"></script>'
        '<style>'
        'html,body{margin:0;padding:0;background:transparent;height:100%;width:100%;overflow:hidden;}'
        'body>div{height:100%;width:100%;}'
        '.plotly-graph-div,.js-plotly-plot{width:100%!important;height:100%!important;}'
        '</style>'
        f'</head><body>{div}</body></html>'
    )


def render_chart(root: str, rel_path: str, chart_type: str, run_name: Optional[str], theme: str) -> str:
    """Render one chart (sweep or per-run) as standalone Plotly HTML."""
    known = set(SWEEP_CHARTS) | set(REQUEST_CHARTS) | set(PERCENTILE_CHARTS)
    if chart_type not in known:
        raise PerfArchiveError(f'Unknown chart_type: {chart_type}', 400)

    run_dir = resolve_run_dir(root, rel_path)
    if run_dir is None:
        raise PerfArchiveError('Invalid path', 400)

    from evalscope.perf.utils.report import perf_charts
    from evalscope.perf.utils.report.perf_data import RunLoader

    runs = _load_runs(run_dir, with_requests=False)
    if not runs:
        raise PerfArchiveError(f'No perf runs found under: {rel_path}', 404)
    is_emb = is_embedding((runs[0].args or {}).get('api', ''))

    if chart_type in SWEEP_CHARTS:
        builder_name, needs_emb = SWEEP_CHARTS[chart_type]
        builder = getattr(perf_charts, builder_name)
        div = builder(runs, is_emb, theme=theme) if needs_emb else builder(runs, theme=theme)
    else:
        # Per-run charts require a specific run sub-directory.
        if not run_name:
            raise PerfArchiveError('run is required for this chart_type', 400)
        run = _find_run(runs, run_name)
        if run is None:
            raise PerfArchiveError(f'Run not found: {run_name}', 404)
        if chart_type in PERCENTILE_CHARTS:
            charts = perf_charts.build_percentile_chart(run, is_emb, theme=theme)
            div = charts[PERCENTILE_CHARTS[chart_type]]
        else:
            # Per-request charts need the DB rows for just this one run.
            full = RunLoader.load_run(os.path.join(run_dir, run_name), run_name, with_requests=True)
            if full is None:
                raise PerfArchiveError(f'Run not found: {run_name}', 404)
            label = REQUEST_CHARTS[chart_type]
            tabs = perf_charts.build_request_detail_tabs(full, is_emb, theme=theme)
            div = next((t['chart'] for t in tabs if t['label'] == label), '')
        if not div:
            raise PerfArchiveError('Chart not available for this run', 404)

    return wrap_chart_html(div)


def render_compare_chart(root: str, rel_paths: List[str], chart_type: str, theme: str) -> str:
    """Overlay a single sweep metric across multiple perf-run directories."""
    if chart_type not in SWEEP_CHARTS:
        raise PerfArchiveError(f'Unknown chart_type: {chart_type}', 400)

    from evalscope.perf.utils.report import perf_charts

    series = []
    emb_modes = set()
    for rel in rel_paths:
        run_dir = resolve_run_dir(root, rel)
        if run_dir is None:
            raise PerfArchiveError(f'Invalid path: {rel}', 400)
        runs = _load_runs(run_dir, with_requests=False)
        if not runs:
            continue
        first_args = runs[0].args or {}
        emb_modes.add(is_embedding(first_args.get('api', '')))
        model = first_args.get('model', first_args.get('model_id', rel))
        ts = extract_timestamp(rel, run_dir)
        label = f'{model} · {ts}' if ts else model
        series.append((label, runs))
    if not series:
        raise PerfArchiveError('No perf runs found for the given paths', 404)
    if len(emb_modes) > 1:
        raise PerfArchiveError('Cannot compare embedding and LLM runs in the same chart', 400)
    is_emb = emb_modes.pop()

    div = perf_charts.build_compare_chart(series, chart_type, is_embedding=is_emb, theme=theme)
    return wrap_chart_html(div)


# ------------------------------------------------------------------
# History report
# ------------------------------------------------------------------


def ensure_history_report(root: str, rel_path: str) -> str:
    """Return the path to the run's HTML report, generating it lazily if absent."""
    run_dir = resolve_run_dir(root, rel_path)
    if run_dir is None:
        raise PerfArchiveError('Invalid path', 400)

    report_file = os.path.join(run_dir, 'perf_report.html')
    if os.path.isfile(report_file):
        return report_file

    from evalscope.perf.utils.report.generate_report import gen_perf_html_report

    runs = _load_runs(run_dir, with_requests=False)
    if not runs:
        raise PerfArchiveError(f'No perf runs found under: {rel_path}', 404)

    api_type = (runs[0].args or {}).get('api', '')
    out_path = gen_perf_html_report(run_dir, {}, SimpleNamespace(api=api_type))
    if not out_path or not os.path.isfile(out_path):
        raise PerfArchiveError('Failed to generate perf report', 500)
    return out_path
