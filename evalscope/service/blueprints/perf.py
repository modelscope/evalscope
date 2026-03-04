import os
import uuid
from flask import Blueprint, jsonify, request

from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.utils.logger import get_logger

try:
    from ..utils import OUTPUT_DIR, get_log_content, run_perf_wrapper, submit_task, task_store
except ImportError:
    from utils import OUTPUT_DIR, get_log_content, run_perf_wrapper, submit_task, task_store  # type: ignore[no-redef]

logger = get_logger()

bp_perf = Blueprint('perf', __name__, url_prefix='/api/v1/perf')


@bp_perf.route('', methods=['POST'])
def run_performance_test():
    """Submit a performance benchmark task (non-blocking).

    Returns task_id immediately; poll GET /api/v1/perf/status?task_id=<id> for progress.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    required_fields = ['model', 'url']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    task_id = request.headers.get('EvalScope-Task-Id', uuid.uuid4().hex)

    # Default to openai API
    if 'api' not in data:
        data['api'] = 'openai'

    perf_args = PerfArguments.from_dict(data)
    perf_args.no_timestamp = True
    perf_args.outputs_dir = os.path.join(OUTPUT_DIR, task_id)
    perf_args.name = 'perf'

    logger.info(f'[{task_id}] Submitting performance benchmark for model: {perf_args.model}')
    logger.info(f'[{task_id}] URL: {perf_args.url}')

    submit_task(task_id, run_perf_wrapper, perf_args)

    return jsonify({'status': 'submitted', 'message': 'Performance test submitted', 'task_id': task_id})


@bp_perf.route('/status', methods=['GET'])
def get_performance_status():
    """Get the current status of a performance benchmark task.

    Query params:
        task_id (str): the task identifier returned by POST /api/v1/perf
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    task = task_store.get(task_id)
    if task is None:
        return jsonify({'error': f'Task not found: {task_id}'}), 404

    return jsonify({'task_id': task_id, **task})


@bp_perf.route('/log', methods=['GET'])
def get_performance_log():
    """Get performance benchmark log content.

    Query params:
        task_id    (str): the task identifier
        start_line (int): skip this many leading lines (default 0)
    """
    try:
        task_id = request.args.get('task_id')
        start_line = request.args.get('start_line', 0, type=int)

        content = get_log_content(task_id, os.path.join('perf', 'benchmark.log'), start_line)
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f'Failed to get performance log: {str(e)}')
        return jsonify({'error': str(e)}), 500
