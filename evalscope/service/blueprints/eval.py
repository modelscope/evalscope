import os
from flask import Blueprint, jsonify, request

from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.utils.logger import get_logger
from ..utils import OUTPUT_DIR, get_log_content, run_eval_wrapper, run_in_subprocess, task_handler

logger = get_logger()

bp_eval = Blueprint('eval', __name__, url_prefix='/api/v1/eval')


@bp_eval.route('', methods=['POST'])
@task_handler(log_subpath=os.path.join('logs', 'eval_log.log'))
def run_evaluation(task_id: str):
    """Run model evaluation."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    required_fields = ['model', 'datasets', 'api_url']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    # Default to OpenAI API compatible models
    if not data.get('eval_type'):
        data['eval_type'] = EvalType.OPENAI_API

    task_config = TaskConfig.from_dict(data)
    task_config.no_timestamp = True
    task_config.work_dir = os.path.join(OUTPUT_DIR, task_id)

    logger.info(f'[{task_id}] Starting evaluation task for model: {task_config.model}')
    logger.info(f'[{task_id}] Datasets: {task_config.datasets}')

    result = run_in_subprocess(run_eval_wrapper, task_config)

    return jsonify({'status': 'success', 'message': 'Evaluation completed', 'result': result, 'task_id': task_id})


@bp_eval.route('/resume', methods=['POST'])
@task_handler(log_subpath=os.path.join('logs', 'eval_log.log'))
def resume_evaluation(task_id: str):
    """Resume a previously interrupted evaluation."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    required_fields = ['model', 'datasets', 'api_url', 'resume_task_id']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    resume_task_id = data.pop('resume_task_id')
    resume_work_dir = os.path.join(OUTPUT_DIR, resume_task_id)

    if not os.path.isdir(resume_work_dir):
        return jsonify({'error': f'Output directory not found for resume_task_id: {resume_task_id}'}), 404

    # Default to OpenAI API compatible models
    if not data.get('eval_type'):
        data['eval_type'] = EvalType.OPENAI_API

    task_config = TaskConfig.from_dict(data)
    task_config.no_timestamp = True
    task_config.use_cache = resume_work_dir
    task_config.rerun_review = True

    logger.info(f'[{task_id}] Resuming evaluation for resume_task_id: {resume_task_id}')
    logger.info(f'[{task_id}] Model: {task_config.model}, Datasets: {task_config.datasets}')

    result = run_in_subprocess(run_eval_wrapper, task_config)

    return jsonify({
        'status': 'success',
        'message': 'Evaluation resumed and completed',
        'result': result,
        'task_id': task_id,
        'resume_task_id': resume_task_id
    })


@bp_eval.route('/log', methods=['GET'])
def get_evaluation_log():
    """Get evaluation log."""
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
