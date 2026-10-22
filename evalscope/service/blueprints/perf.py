import json
import math
import os

from flask import Blueprint, current_app, jsonify, request, send_file
from tabulate import tabulate

from evalscope.metrics.semantics import resolve_perf_semantics
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.registry import ApiRegistry, DatasetRegistry
from evalscope.perf.sla.sla_run import SLAResult, format_sla_tables
from evalscope.perf.utils.benchmark_util import Metrics
from evalscope.perf.utils.rich_display import EmbeddingResultAnalyzer, LLMResultAnalyzer
from evalscope.service.api_models import (
    DeletePerfRunResponse,
    EvalInvokeResponse,
    ListPerfRunsResponse,
    LogResponse,
    PerfDetailResponse,
    PerfRequestsResponse,
    PerfRunsListResponse,
    ProgressResponse,
    TaskStatusResponse,
)
from evalscope.utils.logger import get_logger
from evalscope.utils.uri_utils import sanitize_endpoint, validate_http_endpoint

from .. import perf_archive
from ..perf_archive import PerfArchiveError
from ..responses import json_response
from ..utils import (
    OUTPUT_DIR,
    TaskStoppedError,
    active_task_ids,
    build_task_info,
    create_log_file,
    finish_task,
    get_log_content,
    run_in_subprocess,
    run_perf_wrapper,
    serialize_result,
    stop_process,
    task_file_path,
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
    if isinstance(result, SLAResult):
        return format_sla_tables(result)
    try:
        is_emb = Metrics.is_embedding_or_rerank(api_type)
        analyzer = EmbeddingResultAnalyzer() if is_emb else LLMResultAnalyzer()
        analysis = analyzer.analyze(result)
        if not analysis.rows:
            return ''
        if is_emb:
            headers = [
                '并发数',
                '请求速率',
                '每秒请求数',
                '平均延迟(s)',
                'P99延迟(s)',
                '平均输入TPS',
                'P99输入TPS',
                '平均输入Token数',
                '成功率',
            ]
        else:
            headers = [
                '并发数',
                '请求速率',
                '请求数',
                '每秒请求数',
                '平均延迟(s)',
                'P99延迟(s)',
                '平均首字延迟(s)',
                'P99首字延迟(s)',
                '平均每Token延迟(s)',
                'P99每Token延迟(s)',
                '生成速度(toks/s)',
                '成功率',
            ]
        return tabulate([list(r.values()) for r in analysis.rows], headers=headers, tablefmt='pipe')
    except Exception as e:
        logger.warning(f'Failed to build perf table: {e}')
        return ''


bp_perf = Blueprint('perf', __name__, url_prefix='/api/v1/perf')


def _send_safe_perf_report(report_file: str):
    from evalscope.perf.utils.report.generate_report import is_current_perf_report

    return send_file(report_file, mimetype='text/html', as_attachment=not is_current_perf_report(report_file))


def _validate_performance_config(args: PerfArguments) -> None:
    plugin = ApiRegistry.get_class(args.api)
    if plugin.process_request is ApiPluginBase.process_request:
        raise ValueError('This API protocol does not implement HTTP benchmarking; use a supported protocol')
    DatasetRegistry.get_class(args.dataset)
    if any(not math.isfinite(rate) or (rate != -1 and rate <= 0) for rate in args.rate):
        raise ValueError('Rate must be -1 (unlimited) or positive')
    if any(number <= 0 for number in args.number):
        raise ValueError('Request counts must be positive integers')
    if args.min_tokens is not None and args.min_tokens < 0:
        raise ValueError('Min Tokens cannot be negative')
    if isinstance(args.max_tokens, int):
        if args.max_tokens <= 0:
            raise ValueError('Max Tokens must be positive')
        if args.min_tokens is not None and args.min_tokens > args.max_tokens:
            raise ValueError('Min Tokens cannot exceed Max Tokens')
    if args.min_prompt_length < 0 or args.max_prompt_length < args.min_prompt_length:
        raise ValueError('Invalid prompt length range')
    if args.dataset == 'random' and not args.tokenizer_path:
        raise ValueError('The random dataset requires tokenizer_path')


@bp_perf.route('/invoke', methods=['POST'])
def run_performance_test():
    """Run a performance benchmark task (blocking).

    Returns the benchmark result when the task completes.
    """
    data = request.get_json()
    if not isinstance(data, dict) or not data:
        return jsonify({'error': 'Request body must be a JSON object'}), 400

    required_fields = ['model']
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

    requires_url = data['api'] not in ('local', 'local_vllm') or bool(data.get('url'))
    if requires_url:
        try:
            validate_http_endpoint(data.get('url'))
        except ValueError as error:
            return jsonify({'error': str(error)}), 400
        data['url'] = data['url'].strip()

    try:
        perf_args = PerfArguments.from_dict(data)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid performance configuration'}), 400
    try:
        _validate_performance_config(perf_args)
    except ValueError as error:
        return jsonify({'error': str(error)}), 400

    perf_args.no_timestamp = True
    perf_args.outputs_dir = os.path.join(OUTPUT_DIR, task_id)
    perf_args.name = 'perf'
    perf_args.enable_progress_tracker = True
    perf_args.no_test_connection = True

    logger.info(f'[{task_id}] Running performance benchmark for model: {perf_args.model}')
    logger.info(f'[{task_id}] Endpoint: {sanitize_endpoint(perf_args.url)}')

    create_log_file(task_id, os.path.join('perf', 'benchmark.log'))

    task_meta = build_task_info(task_id, 'perf', perf_args, perf_args.alias)
    try:
        result = run_in_subprocess(run_perf_wrapper, perf_args, task_id=task_id, task_meta=task_meta)
        table_str = _build_perf_table(result, api_type=perf_args.api)
        finish_task(task_id, 'completed')
        logger.info(f'[{task_id}] Task completed successfully')
        return json_response(
            EvalInvokeResponse,
            {'status': 'completed', 'task_id': task_id, 'result': serialize_result(result), 'table': table_str},
        )
    except TaskStoppedError:
        finish_task(task_id, 'stopped')
        logger.info(f'[{task_id}] Task stopped by user.')
        return json_response(EvalInvokeResponse, {'status': 'stopped', 'task_id': task_id})
    except Exception as e:
        finish_task(task_id, 'error')
        logger.error(f'[{task_id}] Task failed: {e}')
        return jsonify({'status': 'error', 'task_id': task_id, 'error': 'Task failed; check the task log'}), 500


@bp_perf.route('/stop', methods=['POST'])
def stop_performance_test():
    """Stop a running performance benchmark task.

    Query params:
        task_id (str): the task identifier
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    try:
        validate_task_id(task_id)
        stopped = stop_process(task_id)
    except ValueError:
        return jsonify({'error': 'Invalid task_id'}), 400
    if stopped:
        return json_response(TaskStatusResponse, {'status': 'stopped', 'task_id': task_id})
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

    try:
        report_file = perf_archive.ensure_history_report(OUTPUT_DIR, os.path.join(task_id, 'perf'))
        return _send_safe_perf_report(report_file)
    except PerfArchiveError as e:
        return jsonify({'error': e.message}), e.status


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
        return json_response(LogResponse, result)
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

    try:
        validate_task_id(task_id)
    except ValueError:
        return jsonify({'error': 'Invalid task_id'}), 400
    try:
        progress_file = task_file_path(task_id, os.path.join('perf', 'progress.json'))
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        return json_response(ProgressResponse, progress)
    except ValueError:
        return jsonify({'error': 'Invalid progress data or path'}), 400
    except FileNotFoundError:
        return json_response(ProgressResponse, {'percent': 0.0})
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
        # The run list exposes its numbers under stable API paths, so the semantics map is keyed
        # by those paths. The run objects themselves are unchanged.
        semantics = resolve_perf_semantics(('best_rps', 'best_latency', 'success_rate'))
        return json_response(
            ListPerfRunsResponse,
            {
                'runs': runs,
                'total': len(runs),
                'metric_semantics': semantics,
            },
        )
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
        return json_response(PerfDetailResponse, perf_archive.build_run_detail(_root_path(), rel_path))
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
        return json_response(PerfRunsListResponse, {'runs': items, 'total': len(items)})
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
        return json_response(PerfRequestsResponse, result)
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
        return _send_safe_perf_report(report_file)
    except PerfArchiveError as e:
        return jsonify({'error': e.message}), e.status
    except Exception as e:
        logger.error(f'Failed to serve perf history report for {rel_path}: {e}')
        return jsonify({'error': str(e)}), 500


@bp_perf.route('/run', methods=['DELETE'])
def delete_perf_run():
    """Delete a historical perf-run directory under the output root.

    Query params:
        root_path (str): output root directory (optional; falls back to config)
        path      (str): run directory path relative to root

    Returns 409 when the run belongs to a task that is still executing.
    """
    rel_path = request.args.get('path')
    if not rel_path:
        return jsonify({'error': 'path is required'}), 400

    # Running-task protection: in the service layout the run path is
    # ``<task_id>/perf``, so refuse deletion while that task is still active.
    segments = set(rel_path.replace('\\', '/').strip('/').split('/'))
    running = segments & active_task_ids()
    if running:
        return jsonify({'error': f'Task is still running: {sorted(running)[0]}'}), 409

    try:
        perf_archive.delete_run(_root_path(), rel_path)
        return json_response(DeletePerfRunResponse, {'success': True, 'path': rel_path})
    except PerfArchiveError as e:
        return jsonify({'error': e.message}), e.status
    except Exception as e:
        logger.error(f'Failed to delete perf run {rel_path}: {e}')
        return jsonify({'error': str(e)}), 500
