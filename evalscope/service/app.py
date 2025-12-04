# Copyright (c) Alibaba, Inc. and its affiliates.
"""Flask service for EvalScope evaluation and performance testing."""
import traceback
from datetime import datetime
from flask import Flask, jsonify, request

from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

app = Flask(__name__)


def create_app():
    """Create and configure the Flask application."""
    return app


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'evalscope', 'timestamp': datetime.now().isoformat()})


@app.route('/api/v1/eval', methods=['POST'])
def run_evaluation():
    """
    Run model evaluation.

    Request body should contain TaskConfig parameters for OpenAI API compatible models.

    Example:
    {
        "model": "qwen-plus",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "your-api-key",
        "datasets": ["gsm8k", "mmlu"],
        "limit": 10,
        "generation_config": {
            "temperature": 0.0,
            "max_tokens": 2048
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        # Validate required parameters
        if 'model' not in data:
            return jsonify({'error': 'model is required'}), 400

        if 'datasets' not in data:
            return jsonify({'error': 'datasets is required'}), 400

        # This service only supports OpenAI API compatible models, so eval_type is always 'service'.
        data['eval_type'] = EvalType.SERVICE

        # Validate that api_url is provided for service-based evaluation
        if 'api_url' not in data:
            return jsonify({'error': 'api_url is required for service-based evaluation'}), 400

        # Create TaskConfig
        task_config = TaskConfig.from_dict(data)

        logger.info(f'Starting evaluation task for model: {task_config.model}')
        logger.info(f'Datasets: {task_config.datasets}')

        # Run evaluation
        result = run_task(task_config)

        return jsonify({
            'status': 'success',
            'message': 'Evaluation completed',
            'result': result,
            'output_dir': task_config.work_dir
        })

    except Exception as e:
        logger.error(f'Evaluation failed: {str(e)}')
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/v1/perf', methods=['POST'])
def run_performance_test():
    """
    Run performance benchmark.

    Request body should contain PerfArguments parameters for OpenAI API compatible models.

    Example:
    {
        "model": "qwen-plus",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api": "openai",
        "api_key": "your-api-key",
        "number": 100,
        "parallel": 10,
        "dataset": "openqa",
        "max_tokens": 2048,
        "temperature": 0.0
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        # Validate required parameters
        if 'model' not in data:
            return jsonify({'error': 'model is required'}), 400

        if 'url' not in data:
            return jsonify({'error': 'url is required'}), 400

        # Set default api to openai if not provided
        if 'api' not in data:
            data['api'] = 'openai'

        # Validate api type
        valid_apis = ['openai', 'dashscope', 'anthropic', 'gemini']
        if data.get('api') not in valid_apis:
            return jsonify({
                'error':
                f'This service only supports OpenAI API compatible models. Valid api types: {valid_apis}'
            }), 400

        # Create PerfArguments
        perf_args = PerfArguments.from_dict(data)

        logger.info(f'Starting performance benchmark for model: {perf_args.model}')
        logger.info(f'URL: {perf_args.url}')
        logger.info(f'Number: {perf_args.number}, Parallel: {perf_args.parallel}')

        # Run performance test
        result = run_perf_benchmark(perf_args)

        # Build response with unified format
        response_data = {
            'status': 'success',
            'message': 'Performance test completed',
            'output_dir': perf_args.outputs_dir,
            'results': result
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f'Performance test failed: {str(e)}')
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/v1/eval/params', methods=['GET'])
def get_eval_params():
    """Get available parameters for evaluation endpoint."""
    params_info = {
        'required': {
            'model': 'str - Model name or identifier',
            'datasets': 'List[str] - List of dataset names to evaluate'
        },
        'optional': {
            'model_id': 'str - Unique identifier for the model',
            'model_args': 'Dict - Additional model initialization arguments',
            'api_url': 'str - API endpoint URL (required for OpenAI API models)',
            'api_key': 'str - API authentication key (default: "EMPTY")',
            'eval_type': 'str - Evaluation type (auto-set to "service" when api_url provided)',
            'dataset_args': 'Dict - Additional dataset loading arguments',
            'dataset_dir': 'str - Directory for cached datasets',
            'limit': 'int|float - Max samples to evaluate (int=count, float=fraction)',
            'eval_batch_size': 'int - Batch size for evaluation (default: 1)',
            'generation_config': {
                'temperature': 'float - Sampling temperature (default: 0.0)',
                'max_tokens': 'int - Maximum tokens to generate (default: 2048)',
                'top_p': 'float - Nucleus sampling parameter',
                'top_k': 'int - Top-k sampling parameter',
                'stream': 'bool - Whether to use streaming',
                'timeout': 'float - Request timeout in seconds'
            },
            'repeats': 'int - Number of times to repeat dataset items (default: 1)',
            'use_cache': 'str - Path to cached results to reuse',
            'work_dir': 'str - Working directory for outputs',
            'debug': 'bool - Enable debug mode (default: false)',
            'seed': 'int - Random seed (default: 42)',
            'judge_strategy': 'str - LLM judge strategy (auto/single/pairwise)',
            'judge_worker_num': 'int - Number of judge workers (default: 1)',
            'judge_model_args': 'Dict - Judge model configuration'
        },
        'example': {
            'model': 'qwen-plus',
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': 'your-api-key',
            'datasets': ['gsm8k', 'mmlu'],
            'limit': 10,
            'generation_config': {
                'temperature': 0.0,
                'max_tokens': 2048
            }
        }
    }
    return jsonify(params_info)


@app.route('/api/v1/perf/params', methods=['GET'])
def get_perf_params():
    """Get available parameters for performance test endpoint."""
    params_info = {
        'required': {
            'model': 'str - Model name or identifier',
            'url': 'str - API endpoint URL'
        },
        'optional': {
            'api': 'str - API type (openai/dashscope/anthropic/gemini, default: "openai")',
            'api_key': 'str - API authentication key',
            'number': 'int|List[int] - Number of requests (default: 1000)',
            'parallel': 'int|List[int] - Number of parallel requests (default: 1)',
            'rate': 'int - Requests per second limit (default: -1, no limit)',
            'dataset': 'str - Dataset to use (default: "openqa")',
            'dataset_path': 'str - Path to custom dataset file',
            'max_tokens': 'int - Maximum tokens to generate (default: 2048)',
            'temperature': 'float - Sampling temperature (default: 0.0)',
            'top_p': 'float - Nucleus sampling parameter',
            'top_k': 'int - Top-k sampling parameter',
            'stream': 'bool - Whether to use streaming (default: true)',
            'max_prompt_length': 'int - Maximum prompt length',
            'min_prompt_length': 'int - Minimum prompt length (default: 0)',
            'prompt': 'str - Custom prompt text',
            'query_template': 'str - Template for query formatting',
            'connect_timeout': 'int - Connection timeout in seconds',
            'read_timeout': 'int - Read timeout in seconds',
            'total_timeout': 'int - Total timeout in seconds (default: 21600)',
            'debug': 'bool - Enable debug mode (default: false)',
            'log_every_n_query': 'int - Log frequency (default: 10)',
            'outputs_dir': 'str - Output directory (default: "outputs")',
            'seed': 'int - Random seed',
            'visualizer': 'str - Visualizer to use (wandb/swanlab/clearml)',
            'name': 'str - Run name for visualizer and output'
        },
        'example': {
            'model': 'qwen-plus',
            'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            'api': 'openai',
            'api_key': 'your-api-key',
            'number': 100,
            'parallel': 10,
            'dataset': 'openqa',
            'max_tokens': 2048,
            'temperature': 0.0
        }
    }
    return jsonify(params_info)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': {
            'GET /health': 'Health check',
            'POST /api/v1/eval': 'Run model evaluation',
            'POST /api/v1/perf': 'Run performance test',
            'GET /api/v1/eval/params': 'Get evaluation parameters info',
            'GET /api/v1/perf/params': 'Get performance test parameters info'
        }
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f'Internal server error: {str(error)}')
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500


def run_service(host: str = '0.0.0.0', port: int = 9000, debug: bool = False):
    """
    Run the Flask service.

    Args:
        host: Host address to bind to (default: 0.0.0.0)
        port: Port to listen on (default: 9000)
        debug: Enable Flask debug mode (default: False)
    """
    logger.info(f'Starting EvalScope service on {host}:{port}')
    logger.info('Available endpoints:')
    logger.info('  GET  /health - Health check')
    logger.info('  POST /api/v1/eval - Run model evaluation')
    logger.info('  POST /api/v1/perf - Run performance test')
    logger.info('  GET  /api/v1/eval/params - Get evaluation parameters')
    logger.info('  GET  /api/v1/perf/params - Get performance parameters')

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_service()
