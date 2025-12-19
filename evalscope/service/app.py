# Copyright (c) Alibaba, Inc. and its affiliates.
"""Flask service for EvalScope evaluation and performance testing."""
import multiprocessing
import os
import traceback
import uuid
from datetime import datetime
from flask import Blueprint, Flask, jsonify, request
from functools import wraps

from evalscope.config import TaskConfig
from evalscope.constants import DEFAULT_WORK_DIR, EvalType
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

OUTPUT_DIR = os.getenv('EVALSCOPE_OUTPUT_DIR', DEFAULT_WORK_DIR)


def _process_worker(func, queue, *args, **kwargs):
    """Worker function to run task in a separate process."""
    try:
        result = func(*args, **kwargs)
        queue.put({'status': 'success', 'result': result})
    except Exception as e:
        queue.put({'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()})


def _run_in_subprocess(func, *args, **kwargs):
    """Run a function in a subprocess and return the result."""
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_process_worker, args=(func, queue, *args), kwargs=kwargs)
    p.start()
    p.join()

    if not queue.empty():
        res = queue.get()
        if res['status'] == 'error':
            raise Exception(f"Subprocess error: {res['error']}\n{res.get('traceback', '')}")
        return res['result']
    else:
        raise Exception(f'Subprocess terminated unexpectedly with exit code {p.exitcode}')


def _log_error_to_file(work_dir: str, sub_path: str, error_msg: str):
    """Write error message to log file."""
    try:
        log_file = os.path.join(work_dir, sub_path)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'\n[Error] {datetime.now().isoformat()}\n')
            f.write(error_msg + '\n')
    except Exception as write_err:
        logger.error(f'Failed to write error to log file: {write_err}')


def handle_exceptions(log_subpath: str = 'error.log'):
    """Decorator to handle exceptions in route handlers."""

    def decorator(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            request_id = request.headers.get('X-Fc-Request-Id', uuid.uuid4().hex)

            try:
                return f(*args, request_id=request_id, **kwargs)
            except Exception as e:
                logger.error(f'Request failed: {str(e)}')
                logger.error(traceback.format_exc())

                # Write error to log file if we can determine the work_dir
                # Note: This assumes the handler has set up the directory structure or we use a default
                work_dir = os.path.join(OUTPUT_DIR, request_id)
                _log_error_to_file(work_dir, log_subpath, traceback.format_exc())

                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'request_id': request_id
                }), 500

        return wrapper

    return decorator


def get_log_content(request_id: str, sub_path: str, start_line: int = 0):
    """Helper to read log content."""
    if not request_id:
        raise ValueError('request_id is required')

    log_file = os.path.join(OUTPUT_DIR, request_id, sub_path)
    if not os.path.exists(log_file):
        raise FileNotFoundError(f'Log file not found: {log_file}')

    with open(log_file, 'r', encoding='utf-8') as f:
        if start_line > 0:
            for _ in range(start_line):
                f.readline()
        content = f.read()
    return content


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
    result = _run_in_subprocess(run_task, task_config)

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
    result = _run_in_subprocess(run_perf_benchmark, perf_args)

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
