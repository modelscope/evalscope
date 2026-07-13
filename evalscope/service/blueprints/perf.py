import json
import os
import re
from datetime import datetime
from flask import Blueprint, current_app, jsonify, request, send_file
from tabulate import tabulate
from types import SimpleNamespace
from typing import List, Optional

from evalscope.constants import PLOTLY_CDN_URL
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.perf.utils.benchmark_util import Metrics
from evalscope.perf.utils.rich_display import EmbeddingResultAnalyzer, LLMResultAnalyzer
from evalscope.utils.logger import get_logger
from ..utils import (
    OUTPUT_DIR,
    create_log_file,
    get_log_content,
    run_in_subprocess,
    run_perf_wrapper,
    serialize_result,
    stop_process,
    validate_task_id,
)

logger = get_logger()

# Recognized perf run-directory markers (mirror perf_data.py)
_PARALLEL_RE = re.compile(r'^parallel_(\d+)_number_(\d+)$')
_RATE_RE = re.compile(r'^rate_([\d.]+)_number_(\d+)$')
# Bound recursive scan depth so large output trees stay cheap to walk.
_MAX_SCAN_DEPTH = 3


def _root_path() -> str:
    """Resolve the outputs root: query param > app config > OUTPUT_DIR."""
    return request.args.get('root_path', current_app.config.get('OUTPUTS_ROOT') or OUTPUT_DIR)


def _is_run_dir(entry_path: str) -> bool:
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


def _scan_perf_runs(root: str) -> List[str]:
    """Recursively discover perf-run directories under *root* (bounded depth).

    Returns a sorted (desc) list of paths relative to *root*. Covers both the
    CLI layout (``<ts>/<model>/``) and the service layout (``<task_id>/perf/``)
    because both share the same marker files.
    """
    if not root or not os.path.isdir(root):
        return []

    root = os.path.abspath(root)
    found: List[str] = []

    def _walk(current: str, depth: int) -> None:
        if depth > _MAX_SCAN_DEPTH:
            return
        try:
            entries = sorted(os.listdir(current))
        except OSError:
            return
        for name in entries:
            entry_path = os.path.join(current, name)
            if not os.path.isdir(entry_path):
                continue
            if _is_run_dir(entry_path):
                found.append(os.path.relpath(entry_path, root))
                # Do not descend further: its children are individual runs.
                continue
            _walk(entry_path, depth + 1)

    _walk(root, 0)
    return sorted(found, reverse=True)


def _extract_timestamp(rel_path: str, abs_path: str) -> str:
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


def _resolve_run_dir(root: str, rel_path: str) -> Optional[str]:
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


def _load_runs(run_dir: str):
    """Load all runs from *run_dir* via RunLoader (lazy import to avoid cycles)."""
    from evalscope.perf.utils.report.perf_data import RunLoader
    return RunLoader.load_all(run_dir)


def _build_run_summary(rel_path: str, abs_path: str) -> Optional[dict]:
    """Build lightweight list-item metadata for one perf-run directory."""
    runs = _load_runs(abs_path)
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
            'has_html': has_html,
            'timestamp': _extract_timestamp(rel_path, abs_path),
        }

    first_args = runs[0].args or {}
    total_requests = sum(r.summary.total_requests for r in runs)
    total_succeed = sum(r.summary.succeed_requests for r in runs)
    success_rate = round(total_succeed / total_requests * 100, 1) if total_requests else 0.0
    best_rps = max(r.summary.request_throughput for r in runs)
    valid_lat = [r.summary.avg_latency for r in runs if r.summary.avg_latency >= 0]
    best_latency = min(valid_lat) if valid_lat else 0.0

    return {
        'path': rel_path,
        'model': first_args.get('model', first_args.get('model_id', 'N/A')),
        'api_type': first_args.get('api', ''),
        'dataset': first_args.get('dataset', ''),
        'num_runs': len(runs),
        'total_requests': total_requests,
        'success_rate': success_rate,
        'best_rps': round(best_rps, 4),
        'best_latency': round(best_latency, 4),
        'has_html': has_html,
        'timestamp': _extract_timestamp(rel_path, abs_path),
    }


def _build_perf_table(result, api_type: str = None) -> str:
    """Build a Markdown pipe-table from perf benchmark results with Chinese headers.

    Returns an empty string when no valid results are found.
    """
    try:
        is_emb = Metrics.is_embedding_or_rerank(api_type)
        analyzer = EmbeddingResultAnalyzer() if is_emb else LLMResultAnalyzer()
        analysis = analyzer.analyze(result)
        if not analysis.rows:
            return ''
        if is_emb:
            headers = ['并发数', '请求速率', '每秒请求数', '平均延迟(s)', 'P99延迟(s)', '平均输入TPS', 'P99输入TPS', '平均输入Token数', '成功率']
        else:
            headers = [
                '并发数', '请求速率', '请求数', '每秒请求数', '平均延迟(s)', 'P99延迟(s)', '平均首字延迟(s)', 'P99首字延迟(s)', '平均每Token延迟(s)',
                'P99每Token延迟(s)', '生成速度(toks/s)', '成功率'
            ]
        return tabulate([list(r.values()) for r in analysis.rows], headers=headers, tablefmt='pipe')
    except Exception as e:
        logger.warning(f'Failed to build perf table: {e}')
        return ''


bp_perf = Blueprint('perf', __name__, url_prefix='/api/v1/perf')


@bp_perf.route('/invoke', methods=['POST'])
def run_performance_test():
    """Run a performance benchmark task (blocking).

    Returns the benchmark result when the task completes.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    required_fields = ['model', 'url']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    task_id = request.headers.get('EvalScope-Task-Id')
    if not task_id:
        return jsonify({'error': 'EvalScope-Task-Id header is required'}), 400
    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Default to openai API
    if 'api' not in data:
        data['api'] = 'openai'

    perf_args = PerfArguments.from_dict(data)
    perf_args.no_timestamp = True
    perf_args.outputs_dir = os.path.join(OUTPUT_DIR, task_id)
    perf_args.name = 'perf'
    perf_args.enable_progress_tracker = True
    perf_args.no_test_connection = True

    logger.info(f'[{task_id}] Running performance benchmark for model: {perf_args.model}')
    logger.info(f'[{task_id}] URL: {perf_args.url}')

    create_log_file(task_id, os.path.join('perf', 'benchmark.log'))

    try:
        result = run_in_subprocess(run_perf_wrapper, perf_args, task_id=task_id)
        table_str = _build_perf_table(result, api_type=perf_args.api)
        logger.info(f'[{task_id}] Task completed successfully')
        return jsonify({
            'status': 'completed',
            'task_id': task_id,
            'result': serialize_result(result),
            'table': table_str
        })
    except Exception as e:
        logger.error(f'[{task_id}] Task failed: {e}')
        return jsonify({'status': 'error', 'task_id': task_id, 'error': str(e)}), 500


@bp_perf.route('/stop', methods=['POST'])
def stop_performance_test():
    """Stop a running performance benchmark task.

    Query params:
        task_id (str): the task identifier
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    stopped = stop_process(task_id)
    if stopped:
        return jsonify({'status': 'stopped', 'task_id': task_id}), 200
    else:
        return jsonify({'error': f'No running task found for task_id: {task_id}'}), 404


@bp_perf.route('/report', methods=['GET'])
def get_performance_report():
    """Get the HTML performance report for a completed task.

    Query params:
        task_id (str): the task identifier
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    report_file = os.path.join(OUTPUT_DIR, task_id, 'perf', 'perf_report.html')
    if not os.path.exists(report_file):
        return jsonify({'error': f'Report not found for task_id: {task_id}'}), 404

    return send_file(report_file, mimetype='text/html')


@bp_perf.route('/log', methods=['GET'])
def get_performance_log():
    """Get performance benchmark log content with pagination.

    Query params:
        task_id    (str): the task identifier
        start_line (int, optional): if not provided, read last `page` lines from end
        page       (int): number of lines to read (default 500)

    Returns:
        dict with text, head_line, tail_line, total_lines
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    start_line = request.args.get('start_line', type=int)
    page = request.args.get('page', 500, type=int)

    try:
        result = get_log_content(task_id, os.path.join('perf', 'benchmark.log'), start_line, page)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f'Failed to get performance log: {str(e)}')
        return jsonify({'error': str(e)}), 500


@bp_perf.route('/progress', methods=['GET'])
def get_performance_progress():
    """Get the real-time hierarchical progress of a running perf benchmark task.

    Query params:
        task_id (str): the task identifier
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    progress_file = os.path.join(OUTPUT_DIR, task_id, 'perf', 'progress.json')
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        return jsonify(progress), 200
    except FileNotFoundError:
        return jsonify({'percent': 0.0}), 200
    except Exception as e:
        logger.error(f'Failed to get progress for task {task_id}: {e}')
        return jsonify({'error': str(e)}), 500


# ------------------------------------------------------------------
# Historical perf-run archive
# ------------------------------------------------------------------


@bp_perf.route('/list', methods=['GET'])
def list_perf_runs():
    """List historical performance-benchmark runs discovered under the output root.

    Query params:
        root_path (str): output root directory (optional; falls back to config)
    """
    try:
        root = _root_path()
        if not root or not os.path.isdir(root):
            return jsonify({'error': 'root_path is required and must be an existing directory'}), 400

        root_abs = os.path.abspath(root)
        runs = []
        for rel_path in _scan_perf_runs(root_abs):
            meta = _build_run_summary(rel_path, os.path.join(root_abs, rel_path))
            if meta is not None:
                runs.append(meta)

        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify({'runs': runs, 'total': len(runs)}), 200
    except Exception as e:
        logger.error(f'Failed to list perf runs: {e}')
        return jsonify({'error': str(e)}), 500


@bp_perf.route('/detail', methods=['GET'])
def get_perf_detail():
    """Return native-render metadata for a single perf-run directory.

    Query params:
        root_path (str): output root directory
        path      (str): run directory path relative to root
    """
    rel_path = request.args.get('path')
    if not rel_path:
        return jsonify({'error': 'path is required'}), 400

    run_dir = _resolve_run_dir(_root_path(), rel_path)
    if run_dir is None:
        return jsonify({'error': 'Invalid path'}), 400

    try:
        from evalscope.perf.utils.report.generate_report import (
            _build_basic_info,
            _build_best_config,
            _build_recommendations,
            _build_summary_table,
            _is_embedding,
        )

        runs = _load_runs(run_dir)
        if not runs:
            return jsonify({'error': f'No perf runs found under: {rel_path}'}), 404

        first_args = runs[0].args or {}
        api_type = first_args.get('api', '')
        is_emb = _is_embedding(api_type)
        summary_columns, summary_rows = _build_summary_table(runs, is_emb)

        return jsonify({
            'path': rel_path,
            'model': first_args.get('model', first_args.get('model_id', 'N/A')),
            'api_type': api_type,
            'dataset': first_args.get('dataset', 'N/A'),
            'generated_at': _extract_timestamp(rel_path, run_dir),
            'basic_info': dict(_build_basic_info(first_args, runs, is_emb)),
            'summary_columns': summary_columns,
            'summary_rows': summary_rows,
            'best_config': dict(_build_best_config(runs)),
            'recommendations': _build_recommendations(runs),
            'num_runs': len(runs),
            'is_embedding': is_emb,
            'has_html': os.path.isfile(os.path.join(run_dir, 'perf_report.html')),
        }), 200
    except Exception as e:
        logger.error(f'Failed to load perf detail for {rel_path}: {e}')
        return jsonify({'error': str(e)}), 500


# chart_type -> (builder attr, whether it needs the is_embedding flag)
_CHART_BUILDERS = {
    'latency': ('build_latency_chart', False),
    'ttft': ('build_ttft_chart', False),
    'tpot': ('build_tpot_chart', False),
    'rps': ('build_rps_chart', False),
    'throughput': ('build_throughput_chart', True),
    'success': ('build_success_chart', False),
}


@bp_perf.route('/chart', methods=['GET'])
def get_perf_chart():
    """Render a single sweep chart as standalone Plotly HTML for a perf run.

    Query params:
        root_path  (str): output root directory
        path       (str): run directory path relative to root
        chart_type (str): latency | ttft | tpot | rps | throughput | success
    """
    rel_path = request.args.get('path')
    chart_type = request.args.get('chart_type', 'latency')
    if not rel_path:
        return jsonify({'error': 'path is required'}), 400
    if chart_type not in _CHART_BUILDERS:
        return jsonify({'error': f'Unknown chart_type: {chart_type}'}), 400

    run_dir = _resolve_run_dir(_root_path(), rel_path)
    if run_dir is None:
        return jsonify({'error': 'Invalid path'}), 400

    try:
        from evalscope.perf.utils.report import perf_charts
        from evalscope.perf.utils.report.generate_report import _is_embedding

        runs = _load_runs(run_dir)
        if not runs:
            return jsonify({'error': f'No perf runs found under: {rel_path}'}), 404

        is_emb = _is_embedding(runs[0].args.get('api', ''))
        builder_name, needs_emb = _CHART_BUILDERS[chart_type]
        builder = getattr(perf_charts, builder_name)
        div = builder(runs, is_emb) if needs_emb else builder(runs)

        html = (
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<script src="{PLOTLY_CDN_URL}" charset="utf-8"></script>'
            '<style>html,body{margin:0;padding:0;background:transparent;}</style>'
            f'</head><body>{div}</body></html>'
        )
        return html, 200, {'Content-Type': 'text/html'}
    except Exception as e:
        logger.error(f'Failed to render perf chart {chart_type} for {rel_path}: {e}')
        return jsonify({'error': str(e)}), 500


@bp_perf.route('/history/report', methods=['GET'])
def get_perf_history_report():
    """Serve (or lazily generate) the full HTML report for a historical perf run.

    Query params:
        root_path (str): output root directory
        path      (str): run directory path relative to root
    """
    rel_path = request.args.get('path')
    if not rel_path:
        return jsonify({'error': 'path is required'}), 400

    run_dir = _resolve_run_dir(_root_path(), rel_path)
    if run_dir is None:
        return jsonify({'error': 'Invalid path'}), 400

    report_file = os.path.join(run_dir, 'perf_report.html')
    if os.path.isfile(report_file):
        return send_file(report_file, mimetype='text/html')

    try:
        from evalscope.perf.utils.report.generate_report import gen_perf_html_report

        runs = _load_runs(run_dir)
        if not runs:
            return jsonify({'error': f'No perf runs found under: {rel_path}'}), 404

        api_type = runs[0].args.get('api', '')
        out_path = gen_perf_html_report(run_dir, {}, SimpleNamespace(api=api_type))
        if not out_path or not os.path.isfile(out_path):
            return jsonify({'error': 'Failed to generate perf report'}), 500
        return send_file(out_path, mimetype='text/html')
    except Exception as e:
        logger.error(f'Failed to serve perf history report for {rel_path}: {e}')
        return jsonify({'error': str(e)}), 500
