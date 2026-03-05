import os
from flask import Blueprint, jsonify, request

from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.utils.logger import get_logger

try:
    from ..utils import OUTPUT_DIR, create_log_file, get_log_content, run_in_subprocess, run_perf_wrapper
except ImportError:
    from utils import (  # type: ignore[no-redef]
        OUTPUT_DIR,
        create_log_file,
        get_log_content,
        run_in_subprocess,
        run_perf_wrapper,
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

    required_fields = ['model', 'url']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    task_id = request.headers.get('EvalScope-Task-Id')
    if not task_id:
        return jsonify({'error': 'EvalScope-Task-Id header is required'}), 400

    # Default to openai API
    if 'api' not in data:
        data['api'] = 'openai'

    perf_args = PerfArguments.from_dict(data)
    perf_args.no_timestamp = True
    perf_args.outputs_dir = os.path.join(OUTPUT_DIR, task_id)
    perf_args.name = 'perf'

    logger.info(f'[{task_id}] Running performance benchmark for model: {perf_args.model}')
    logger.info(f'[{task_id}] URL: {perf_args.url}')

    create_log_file(task_id, os.path.join('perf', 'benchmark.log'))

    try:
        result = run_in_subprocess(run_perf_wrapper, perf_args)
        logger.info(f'[{task_id}] Task completed successfully')
        return jsonify({'status': 'completed', 'task_id': task_id, 'result': result})
    except Exception as e:
        logger.error(f'[{task_id}] Task failed: {e}')
        return jsonify({'status': 'error', 'task_id': task_id, 'error': str(e)}), 500


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
