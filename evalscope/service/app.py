# Copyright (c) Alibaba, Inc. and its affiliates.
"""Flask service for EvalScope evaluation and performance testing."""
from datetime import datetime
from flask import Flask, jsonify

from evalscope.utils.logger import get_logger
from .blueprints import bp_eval, bp_perf

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

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint not found',
            'available_endpoints': {
                'GET /health': 'Health check',
                'POST /api/v1/eval': 'Run model evaluation',
                'GET /api/v1/eval/log': 'Get evaluation log',
                'POST /api/v1/eval/resume': 'Resume a previous evaluation',
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
    logger.info('  POST /api/v1/eval/resume - Resume a previous evaluation')
    logger.info('  POST /api/v1/perf - Run performance test')
    logger.info('  GET  /api/v1/perf/log - Get performance test log')
    logger.info('Refer to docs for parameters: https://evalscope.readthedocs.io/en/latest/user_guides/service.html')

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_service()
