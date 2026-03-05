import os
from flask import Blueprint, jsonify, request

from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.utils.logger import get_logger

try:
    from ..utils import OUTPUT_DIR, create_log_file, get_log_content, run_eval_wrapper, run_in_subprocess
except ImportError:
    from utils import (  # type: ignore[no-redef]
        OUTPUT_DIR,
        create_log_file,
        get_log_content,
        run_eval_wrapper,
        run_in_subprocess,
    )

logger = get_logger()

bp_eval = Blueprint('eval', __name__, url_prefix='/api/v1/eval')

_REQUIRED_FIELDS = ['model', 'datasets', 'api_url']


def _parse_request():
    """Validate the request body and return (data, task_id) or a Flask error response.

    Returns:
        (dict, str)  on success.
        Flask response on validation failure.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    for field in _REQUIRED_FIELDS:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    task_id = request.headers.get('EvalScope-Task-Id')
    if not task_id:
        return jsonify({'error': 'EvalScope-Task-Id header is required'}), 400

    return data, task_id


def _build_task_config(data: dict) -> TaskConfig:
    """Build a TaskConfig from request data with common defaults applied."""
    if not data.get('eval_type'):
        data['eval_type'] = EvalType.SERVICE

    task_config = TaskConfig.from_dict(data)
    task_config.no_timestamp = True
    return task_config


def _execute_task(task_id: str, task_config: TaskConfig, label: str = 'Task'):
    """Run the evaluation subprocess and return a Flask response."""
    create_log_file(task_id, os.path.join('logs', 'eval_log.log'))
    try:
        result = run_in_subprocess(run_eval_wrapper, task_config)
        logger.info(f'[{task_id}] {label} completed successfully')
        return jsonify({'status': 'completed', 'task_id': task_id, 'result': result})
    except Exception as e:
        logger.error(f'[{task_id}] {label} failed: {e}')
        return jsonify({'status': 'error', 'task_id': task_id, 'error': str(e)}), 500


@bp_eval.route('/invoke', methods=['POST'])
def run_evaluation():
    """Run a model evaluation task (blocking).

    Returns the evaluation result when the task completes.
    """
    parsed = _parse_request()
    if not isinstance(parsed, tuple) or not isinstance(parsed[0], dict):
        return parsed  # error response
    data, task_id = parsed

    task_config = _build_task_config(data)
    task_config.work_dir = os.path.join(OUTPUT_DIR, task_id)

    logger.info(f'[{task_id}] Running evaluation task for model: {task_config.model}')
    logger.info(f'[{task_id}] Datasets: {task_config.datasets}')

    return _execute_task(task_id, task_config, label='Task')


@bp_eval.route('/resume/invoke', methods=['POST'])
def resume_evaluation():
    """Resume a previously interrupted evaluation task (blocking).

    Returns the evaluation result when the task completes.
    """
    parsed = _parse_request()
    if not isinstance(parsed, tuple) or not isinstance(parsed[0], dict):
        return parsed  # error response
    data, task_id = parsed

    work_dir = os.path.join(OUTPUT_DIR, task_id)
    if not os.path.isdir(work_dir):
        return jsonify({'error': f'Output directory not found for task_id: {task_id}'}), 404

    task_config = _build_task_config(data)
    task_config.use_cache = work_dir
    task_config.rerun_review = True

    logger.info(f'[{task_id}] Running resume task, work_dir: {work_dir}')
    logger.info(f'[{task_id}] Model: {task_config.model}, Datasets: {task_config.datasets}')

    return _execute_task(task_id, task_config, label='Resume task')


@bp_eval.route('/log', methods=['GET'])
def get_evaluation_log():
    """Get evaluation log content.

    Query params:
        task_id    (str): the task identifier
        start_line (int): skip this many leading lines (default 0)
    """
    try:
        task_id = request.args.get('task_id')
        start_line = request.args.get('start_line', 0, type=int)

        content = get_log_content(task_id, os.path.join('logs', 'eval_log.log'), start_line)
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f'Failed to get evaluation log: {str(e)}')
        return jsonify({'error': str(e)}), 500
