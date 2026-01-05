# Copyright (c) Alibaba, Inc. and its affiliates.
"""Flask service for EvalScope evaluation and performance testing."""
import os
from datetime import datetime
from flask import Blueprint, Flask, jsonify, request

from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.utils.logger import get_logger
from .utils import OUTPUT_DIR, get_log_content, handle_exceptions, run_in_subprocess, run_perf_wrapper, run_task_wrapper

logger = get_logger()

# --- Blueprints ---

bp_eval = Blueprint('eval', __name__, url_prefix='/api/v1/eval')
bp_perf = Blueprint('perf', __name__, url_prefix='/api/v1/perf')


@bp_eval.route('', methods=['POST'])
@handle_exceptions(log_subpath=os.path.join('logs', 'eval_log.log'))
def run_evaluation(request_id: str):
    """Run model evaluation."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    # Validate required parameters
    required_fields = ['model', 'datasets', 'api_url']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    # This service only supports OpenAI API compatible models
    data['eval_type'] = EvalType.SERVICE

    # Create TaskConfig
    task_config = TaskConfig.from_dict(data)
    task_config.no_timestamp = True
    task_config.work_dir = os.path.join(OUTPUT_DIR, request_id)

    logger.info(f'[{request_id}] Starting evaluation task for model: {task_config.model}')
    logger.info(f'[{request_id}] Datasets: {task_config.datasets}')

    # Run evaluation
    result = run_in_subprocess(run_task_wrapper, task_config)

    return jsonify({
        'status': 'success',
        'message': 'Evaluation completed',
        'result': result,
        'output_dir': task_config.work_dir,
        'request_id': request_id
    })


@bp_eval.route('/log', methods=['GET'])
def get_evaluation_log():
    """Get evaluation log."""
    try:
        request_id = request.args.get('request_id')
        start_line = request.args.get('start_line', 0, type=int)

        content = get_log_content(request_id, os.path.join('logs', 'eval_log.log'), start_line)
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f'Failed to get evaluation log: {str(e)}')
        return jsonify({'error': str(e)}), 500


@bp_perf.route('', methods=['POST'])
@handle_exceptions(log_subpath=os.path.join('perf', 'benchmark.log'))
def run_performance_test(request_id: str):
    """Run performance benchmark."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    # Validate required parameters
    required_fields = ['model', 'url']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    # Set default api to openai if not provided
    if 'api' not in data:
        data['api'] = 'openai'

    # Validate api type
    valid_apis = ['openai', 'dashscope', 'anthropic', 'gemini']
    if data.get('api') not in valid_apis:
        return jsonify({'error': f'Invalid api type. Valid types: {valid_apis}'}), 400

    # Create PerfArguments
    perf_args = PerfArguments.from_dict(data)
    perf_args.no_timestamp = True
    perf_args.outputs_dir = os.path.join(OUTPUT_DIR, request_id)
    perf_args.name = 'perf'

    logger.info(f'[{request_id}] Starting performance benchmark for model: {perf_args.model}')
    logger.info(f'[{request_id}] URL: {perf_args.url}')

    # Run performance test
    result = run_in_subprocess(run_perf_wrapper, perf_args)

    return jsonify({
        'status': 'success',
        'message': 'Performance test completed',
        'output_dir': perf_args.outputs_dir,
        'results': result,
        'request_id': request_id
    })


@bp_perf.route('/log', methods=['GET'])
def get_performance_log():
    """Get performance test log."""
    try:
        request_id = request.args.get('request_id')
        start_line = request.args.get('start_line', 0, type=int)

        content = get_log_content(request_id, os.path.join('perf', 'benchmark.log'), start_line)
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f'Failed to get performance log: {str(e)}')
        return jsonify({'error': str(e)}), 500


# --- App Factory ---


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(bp_eval)
    app.register_blueprint(bp_perf)

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({'status': 'ok', 'service': 'evalscope', 'timestamp': datetime.now().isoformat()})

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint not found',
            'available_endpoints': {
                'GET /health': 'Health check',
                'POST /api/v1/eval': 'Run model evaluation',
                'GET /api/v1/eval/log': 'Get evaluation log',
                'POST /api/v1/perf': 'Run performance test',
                'GET /api/v1/perf/log': 'Get performance test log'
            }
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f'Internal server error: {str(error)}')
        return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

    return app


def run_service(host: str = '0.0.0.0', port: int = 9000, debug: bool = False):
    """Run the Flask service."""
    app = create_app()

    logger.info(f'Starting EvalScope service on {host}:{port}')
    logger.info('Available endpoints:')
    logger.info('  GET  /health - Health check')
    logger.info('  POST /api/v1/eval - Run model evaluation')
    logger.info('  GET  /api/v1/eval/log - Get evaluation log')
    logger.info('  POST /api/v1/perf - Run performance test')
    logger.info('  GET  /api/v1/perf/log - Get performance test log')
    logger.info('Refer to docs for parameters: https://evalscope.readthedocs.io/en/latest/user_guides/service.html')

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_service()
