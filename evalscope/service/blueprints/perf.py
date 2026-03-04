import os
from flask import Blueprint, jsonify, request

from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.utils.logger import get_logger
from ..utils import OUTPUT_DIR, get_log_content, run_in_subprocess, run_perf_wrapper, task_handler

logger = get_logger()

bp_perf = Blueprint('perf', __name__, url_prefix='/api/v1/perf')


@bp_perf.route('', methods=['POST'])
@task_handler(log_subpath=os.path.join('perf', 'benchmark.log'))
def run_performance_test(task_id: str):
    """Run performance benchmark."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    required_fields = ['model', 'url']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    # Set default api to openai if not provided
    if 'api' not in data:
        data['api'] = 'openai'

    perf_args = PerfArguments.from_dict(data)
    perf_args.no_timestamp = True
    perf_args.outputs_dir = os.path.join(OUTPUT_DIR, task_id)
    perf_args.name = 'perf'

    logger.info(f'[{task_id}] Starting performance benchmark for model: {perf_args.model}')
    logger.info(f'[{task_id}] URL: {perf_args.url}')

    result = run_in_subprocess(run_perf_wrapper, perf_args)

    return jsonify({
        'status': 'success',
        'message': 'Performance test completed',
        'results': result,
        'task_id': task_id
    })


@bp_perf.route('/log', methods=['GET'])
def get_performance_log():
    """Get performance test log."""
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
