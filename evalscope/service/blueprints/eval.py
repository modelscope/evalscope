import os
import uuid
from flask import Blueprint, jsonify, request

from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.utils.logger import get_logger

try:
    from ..utils import OUTPUT_DIR, get_log_content, run_eval_wrapper, run_in_subprocess
except ImportError:
    from utils import OUTPUT_DIR, get_log_content, run_eval_wrapper, run_in_subprocess  # type: ignore[no-redef]

logger = get_logger()

bp_eval = Blueprint('eval', __name__, url_prefix='/api/v1/eval')


@bp_eval.route('/invoke', methods=['POST'])
def run_evaluation():
    """Run a model evaluation task (blocking).

    Returns the evaluation result when the task completes.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    required_fields = ['model', 'datasets', 'api_url']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    task_id = request.headers.get('X-Fc-Async-Task-Id', uuid.uuid4().hex)

    # Default to OpenAI API compatible models
    if not data.get('eval_type'):
        data['eval_type'] = EvalType.SERVICE

    task_config = TaskConfig.from_dict(data)
    task_config.no_timestamp = True
    task_config.work_dir = os.path.join(OUTPUT_DIR, task_id)

    logger.info(f'[{task_id}] Running evaluation task for model: {task_config.model}')
    logger.info(f'[{task_id}] Datasets: {task_config.datasets}')

    try:
        result = run_in_subprocess(run_eval_wrapper, task_config)
        logger.info(f'[{task_id}] Task completed successfully')
        return jsonify({'status': 'completed', 'task_id': task_id, 'result': result})
    except Exception as e:
        logger.error(f'[{task_id}] Task failed: {e}')
        return jsonify({'status': 'error', 'task_id': task_id, 'error': str(e)}), 500


@bp_eval.route('/resume/invoke', methods=['POST'])
def resume_evaluation():
    """Resume a previously interrupted evaluation task (blocking).

    Returns the evaluation result when the task completes.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    required_fields = ['model', 'datasets', 'api_url', 'resume_task_id']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    task_id = request.headers.get('X-Fc-Async-Task-Id', uuid.uuid4().hex)
    resume_task_id = data.pop('resume_task_id')
    resume_work_dir = os.path.join(OUTPUT_DIR, resume_task_id)

    if not os.path.isdir(resume_work_dir):
        return jsonify({'error': f'Output directory not found for resume_task_id: {resume_task_id}'}), 404

    # Default to OpenAI API compatible models
    if not data.get('eval_type'):
        data['eval_type'] = EvalType.SERVICE

    task_config = TaskConfig.from_dict(data)
    task_config.no_timestamp = True
    task_config.use_cache = resume_work_dir
    task_config.rerun_review = True

    logger.info(f'[{task_id}] Running resume task for resume_task_id: {resume_task_id}')
    logger.info(f'[{task_id}] Model: {task_config.model}, Datasets: {task_config.datasets}')

    try:
        result = run_in_subprocess(run_eval_wrapper, task_config)
        logger.info(f'[{task_id}] Resume task completed successfully')
        return jsonify({
            'status': 'completed',
            'task_id': task_id,
            'resume_task_id': resume_task_id,
            'result': result,
        })
    except Exception as e:
        logger.error(f'[{task_id}] Resume task failed: {e}')
        return jsonify({
            'status': 'error',
            'task_id': task_id,
            'resume_task_id': resume_task_id,
            'error': str(e),
        }), 500


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
