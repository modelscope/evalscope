import json
import os
from flask import Blueprint, jsonify, request, send_file
from pydantic import ValidationError

from evalscope.perf import OutputConfig, PerfConfig
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

bp_perf = Blueprint('perf', __name__, url_prefix='/api/v1/perf')


@bp_perf.route('/invoke', methods=['POST'])
def run_performance_test():
    """Run a performance benchmark task (blocking).

    Returns the benchmark result when the task completes.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    required_fields = ['target', 'suite']
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

    try:
        perf_args = PerfConfig.model_validate(data)
    except ValidationError as e:
        return jsonify({'error': 'Invalid performance configuration', 'details': e.errors(include_url=False)}), 400
    perf_args = perf_args.model_copy(
        update={
            'output': OutputConfig(
                root=os.path.join(OUTPUT_DIR, task_id),
                run_id='perf',
                overwrite=True,
                html_report=True,
                console_report=False,
            ),
            'runtime': perf_args.runtime.model_copy(update={'progress': True}),
            'target': perf_args.target.model_copy(update={'skip_connection_test': True}),
        }
    )

    logger.info(f'[{task_id}] Running performance benchmark for model: {perf_args.target.model}')
    logger.info(f'[{task_id}] URL: {perf_args.target.base_url}')

    create_log_file(task_id, os.path.join('perf', 'benchmark.log'))

    try:
        result = run_in_subprocess(run_perf_wrapper, perf_args, task_id=task_id)
        logger.info(f'[{task_id}] Task completed successfully')
        return jsonify({
            'status': 'completed',
            'task_id': task_id,
            'result': serialize_result(result),
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
