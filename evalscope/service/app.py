# Copyright (c) Alibaba, Inc. and its affiliates.
"""Flask service for EvalScope evaluation and performance testing."""
import multiprocessing
from datetime import datetime
from flask import Flask, jsonify

from evalscope.utils.logger import get_logger

# Support two deployment modes:
#   1. Installed package  → relative imports work (evalscope.service.*)
#   2. Flat /code/ copy   → no parent package; pin /code/ first in sys.path
try:
    from .blueprints import bp_eval, bp_perf
except ImportError:
    from blueprints import bp_eval, bp_perf  # type: ignore[no-redef]

logger = get_logger()


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Ensure non-ASCII characters (e.g. Chinese) are serialised as-is in JSON
    # responses instead of being escaped to \uXXXX sequences.
    app.json.ensure_ascii = False

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
                'GET  /health': 'Health check',
                'POST /api/v1/eval/invoke': 'Run model evaluation task (blocking)',
                'GET  /api/v1/eval/benchmarks': 'List supported benchmarks with descriptions',
                'GET  /api/v1/eval/log': 'Get evaluation log',
                'GET  /api/v1/eval/progress': 'Get real-time evaluation progress',
                'GET  /api/v1/eval/report': 'Get HTML evaluation report',
                'POST /api/v1/eval/resume/invoke': 'Resume a previous evaluation (blocking)',
                'POST /api/v1/perf/invoke': 'Run performance benchmark task (blocking)',
                'GET  /api/v1/perf/log': 'Get performance benchmark log',
                'GET  /api/v1/perf/progress': 'Get real-time performance benchmark progress',
                'GET  /api/v1/perf/report': 'Get HTML performance benchmark report'
            }
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f'Internal server error: {str(error)}')
        return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

    return app


def run_service(host: str = '0.0.0.0', port: int = 9000, debug: bool = False):
    """Run the Flask service."""

    # Force the multiprocessing start method to 'spawn' to avoid issues with
    # model loading in forked child processes on some platforms.
    multiprocessing.set_start_method('spawn', force=True)
    app = create_app()

    logger.info(f'Starting EvalScope service on {host}:{port}')
    logger.info('Available endpoints:')
    logger.info('  GET  /health                         - Health check')
    logger.info('  POST /api/v1/eval/invoke             - Run model evaluation task (blocking)')
    logger.info('  GET  /api/v1/eval/benchmarks         - List supported benchmarks with descriptions')
    logger.info('  GET  /api/v1/eval/log                - Get evaluation log')
    logger.info('  GET  /api/v1/eval/progress           - Get real-time evaluation progress')
    logger.info('  GET  /api/v1/eval/report             - Get HTML evaluation report')
    logger.info('  POST /api/v1/eval/resume/invoke      - Resume a previous evaluation (blocking)')
    logger.info('  POST /api/v1/perf/invoke             - Run performance benchmark task (blocking)')
    logger.info('  GET  /api/v1/perf/log                - Get performance benchmark log')
    logger.info('  GET  /api/v1/perf/progress           - Get real-time performance benchmark progress')
    logger.info('  GET  /api/v1/perf/report             - Get HTML performance benchmark report')
    logger.info('Refer to docs for parameters: https://evalscope.readthedocs.io/en/latest/user_guides/service.html')

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_service()
