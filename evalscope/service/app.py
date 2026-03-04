# Copyright (c) Alibaba, Inc. and its affiliates.
"""Flask service for EvalScope evaluation and performance testing."""
from datetime import datetime
from flask import Flask, jsonify

from evalscope.utils.logger import get_logger

# Support two deployment modes:
#   1. Installed package  → relative imports work (evalscope.service.*)
#   2. Flat /code/ copy   → no parent package; sibling dirs are on sys.path
try:
    from .blueprints import bp_eval, bp_perf
    from .utils import task_store
except ImportError:
    from blueprints import bp_eval, bp_perf  # type: ignore[no-redef]
    from utils import task_store  # type: ignore[no-redef]

logger = get_logger()


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

    @app.route('/api/v1/tasks', methods=['GET'])
    def list_tasks():
        """List all submitted tasks and their current status."""
        return jsonify(task_store.all())

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint not found',
            'available_endpoints': {
                'GET  /health': 'Health check',
                'GET  /api/v1/tasks': 'List all tasks and their status',
                'POST /api/v1/eval': 'Submit model evaluation task',
                'GET  /api/v1/eval/status': 'Get evaluation task status',
                'GET  /api/v1/eval/log': 'Get evaluation log',
                'POST /api/v1/eval/resume': 'Resume a previous evaluation',
                'POST /api/v1/perf': 'Submit performance benchmark task',
                'GET  /api/v1/perf/status': 'Get performance task status',
                'GET  /api/v1/perf/log': 'Get performance benchmark log'
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
    logger.info('  GET  /health                  - Health check')
    logger.info('  GET  /api/v1/tasks            - List all tasks and their status')
    logger.info('  POST /api/v1/eval             - Submit model evaluation task')
    logger.info('  GET  /api/v1/eval/status      - Get evaluation task status')
    logger.info('  GET  /api/v1/eval/log         - Get evaluation log')
    logger.info('  POST /api/v1/eval/resume      - Resume a previous evaluation')
    logger.info('  POST /api/v1/perf             - Submit performance benchmark task')
    logger.info('  GET  /api/v1/perf/status      - Get performance task status')
    logger.info('  GET  /api/v1/perf/log         - Get performance benchmark log')
    logger.info('Refer to docs for parameters: https://evalscope.readthedocs.io/en/latest/user_guides/service.html')

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_service()
