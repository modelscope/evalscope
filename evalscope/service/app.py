# Copyright (c) Alibaba, Inc. and its affiliates.
"""Flask service for EvalScope evaluation and performance testing."""
import multiprocessing
import os
from datetime import datetime
from flask import Flask, jsonify, send_from_directory

from evalscope.utils.logger import get_logger
from .blueprints import bp_eval, bp_perf, bp_reports

logger = get_logger()

# Path to the built React SPA (web/dist).  Resolved relative to the
# repository root so that ``pip install -e .`` works out of the box.
_WEB_DIST = os.path.join(os.path.dirname(__file__), '..', 'web', 'dist')
_WEB_DIST = os.path.abspath(_WEB_DIST)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Ensure non-ASCII characters (e.g. Chinese) are serialised as-is in JSON
    # responses instead of being escaped to \uXXXX sequences.
    app.json.ensure_ascii = False

    # --- CORS (development convenience) -----------------------------------
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError:
        pass  # flask-cors not installed; same-origin only

    # Register blueprints
    app.register_blueprint(bp_eval)
    app.register_blueprint(bp_perf)
    app.register_blueprint(bp_reports)

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({'status': 'ok', 'service': 'evalscope', 'timestamp': datetime.now().isoformat()})

    # --- SPA static-file serving ------------------------------------------
    if os.path.isdir(_WEB_DIST):

        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve_spa(path):
            """Serve the React SPA for all non-API routes."""
            file_path = os.path.join(_WEB_DIST, path)
            if path and os.path.isfile(file_path):
                return send_from_directory(_WEB_DIST, path)
            return send_from_directory(_WEB_DIST, 'index.html')

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
                'GET  /api/v1/perf/report': 'Get HTML performance benchmark report',
                'GET  /api/v1/reports/scan': 'Scan available report folders',
                'GET  /api/v1/reports/list': 'Filterable, paginated report listing',
                'GET  /api/v1/reports/load': 'Load a single report',
                'GET  /api/v1/reports/load_multi': 'Load multiple reports',
                'GET  /api/v1/reports/dataframe': 'Get report data as JSON table',
                'GET  /api/v1/reports/predictions': 'Get model predictions',
                'GET  /api/v1/reports/analysis': 'Get AI analysis text',
                'GET  /api/v1/reports/html': 'Get HTML report file',
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
