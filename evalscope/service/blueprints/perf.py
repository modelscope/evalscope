import json
import os
from flask import Blueprint, current_app, jsonify, request, send_file
from tabulate import tabulate

from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.perf.utils.benchmark_util import Metrics
from evalscope.perf.utils.rich_display import EmbeddingResultAnalyzer, LLMResultAnalyzer
from evalscope.utils.logger import get_logger
from .. import perf_archive
from ..perf_archive import PerfArchiveError
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


def _root_path() -> str:
    """Resolve the outputs root: query param > app config > OUTPUT_DIR."""
    return request.args.get('root_path', current_app.config.get('OUTPUTS_ROOT') or OUTPUT_DIR)


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
#
# These endpoints stay thin: request parsing + JSON/error translation only.
# All discovery / view-model / chart / paging logic lives in
# :mod:`evalscope.service.perf_archive`.
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

        runs = perf_archive.list_run_summaries(root)
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
    try:
        return jsonify(perf_archive.build_run_detail(_root_path(), rel_path)), 200
    except PerfArchiveError as e:
        return jsonify({'error': e.message}), e.status
    except Exception as e:
        logger.error(f'Failed to load perf detail for {rel_path}: {e}')
        return jsonify({'error': str(e)}), 500


@bp_perf.route('/chart', methods=['GET'])
def get_perf_chart():
    """Render a single chart as standalone Plotly HTML for a perf run.

    Query params:
        root_path  (str): output root directory
        path       (str): run directory path relative to root
        chart_type (str): sweep (latency|ttft|tpot|rps|throughput|success) or
                          per-run (percentile_latency|percentile_token|
                          req_latency|req_ttft_tpot|req_tokens|req_success)
        run        (str): required for per-run chart types; the run sub-dir name
        theme      (str): 'dark' (default) or 'light'
    """
    rel_path = request.args.get('path')
    chart_type = request.args.get('chart_type', 'latency')
    run_name = request.args.get('run')
    theme = 'light' if request.args.get('theme') == 'light' else 'dark'
    if not rel_path:
        return jsonify({'error': 'path is required'}), 400
    try:
        html = perf_archive.render_chart(_root_path(), rel_path, chart_type, run_name, theme)
        return html, 200, {'Content-Type': 'text/html'}
    except PerfArchiveError as e:
        return jsonify({'error': e.message}), e.status
    except Exception as e:
        logger.error(f'Failed to render perf chart {chart_type} for {rel_path}: {e}')
        return jsonify({'error': str(e)}), 500


@bp_perf.route('/compare/chart', methods=['GET'])
def get_perf_compare_chart():
    """Overlay a single sweep metric across multiple perf-run directories.

    Query params:
        root_path  (str): output root directory
        paths      (str): ';'-separated run directory paths (relative to root)
        chart_type (str): sweep metric (latency|ttft|tpot|rps|throughput|success)
        theme      (str): 'dark' (default) or 'light'
    """
    paths_raw = request.args.get('paths', '')
    chart_type = request.args.get('chart_type', 'rps')
    theme = 'light' if request.args.get('theme') == 'light' else 'dark'
    rel_paths = [p for p in paths_raw.split(';') if p.strip()]
    if not rel_paths:
        return jsonify({'error': 'paths is required'}), 400
    try:
        html = perf_archive.render_compare_chart(_root_path(), rel_paths, chart_type, theme)
        return html, 200, {'Content-Type': 'text/html'}
    except PerfArchiveError as e:
        return jsonify({'error': e.message}), e.status
    except Exception as e:
        logger.error(f'Failed to render perf compare chart {chart_type}: {e}')
        return jsonify({'error': str(e)}), 500


@bp_perf.route('/runs', methods=['GET'])
def list_perf_run_details():
    """List individual runs (parallel_*/rate_*) within a perf-run directory.

    Query params:
        root_path (str): output root directory
        path      (str): run directory path relative to root
    """
    rel_path = request.args.get('path')
    if not rel_path:
        return jsonify({'error': 'path is required'}), 400
    try:
        items = perf_archive.list_run_items(_root_path(), rel_path)
        return jsonify({'runs': items, 'total': len(items)}), 200
    except PerfArchiveError as e:
        return jsonify({'error': e.message}), e.status
    except Exception as e:
        logger.error(f'Failed to list perf run details for {rel_path}: {e}')
        return jsonify({'error': str(e)}), 500


@bp_perf.route('/requests', methods=['GET'])
def get_perf_requests():
    """Return paginated per-request records (from benchmark_data.db) for one run.

    Query params:
        root_path (str): output root directory
        path      (str): run directory path relative to root
        run       (str): the run sub-dir name (parallel_*/rate_*)
        status    (str): optional 'success' | 'failed' filter
        page      (int): 1-based page (default 1)
        page_size (int): rows per page (default 50, max 500)
    """
    rel_path = request.args.get('path')
    run_name = request.args.get('run')
    if not rel_path or not run_name:
        return jsonify({'error': 'path and run are required'}), 400
    try:
        result = perf_archive.query_request_page(
            _root_path(),
            rel_path,
            run_name,
            request.args.get('status'),
            request.args.get('page', 1, type=int),
            request.args.get('page_size', 50, type=int),
        )
        return jsonify(result), 200
    except PerfArchiveError as e:
        return jsonify({'error': e.message}), e.status
    except Exception as e:
        logger.error(f'Failed to load perf requests for {rel_path}/{run_name}: {e}')
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
    try:
        report_file = perf_archive.ensure_history_report(_root_path(), rel_path)
        return send_file(report_file, mimetype='text/html')
    except PerfArchiveError as e:
        return jsonify({'error': e.message}), e.status
    except Exception as e:
        logger.error(f'Failed to serve perf history report for {rel_path}: {e}')
        return jsonify({'error': str(e)}), 500
