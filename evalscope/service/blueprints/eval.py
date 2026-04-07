import json
import os
from flask import Blueprint, current_app, jsonify, request, send_file
from tabulate import tabulate
from typing import Any, Dict, List

from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.report.combinator import get_data_frame, get_report_list
from evalscope.utils.logger import get_logger
from ..utils import (
    DEFAULT_MULTIMODAL_BENCHMARKS,
    DEFAULT_TEXT_BENCHMARKS,
    OUTPUT_DIR,
    build_benchmark_entry,
    create_log_file,
    get_log_content,
    run_eval_wrapper,
    run_in_subprocess,
    validate_task_id,
)

logger = get_logger()

bp_eval = Blueprint('eval', __name__, url_prefix='/api/v1/eval')

_COLUMN_ZH = {
    'Model': '模型',
    'Dataset': '数据集',
    'Metric': '指标',
    'Subset': '子集',
    'Num': '数量',
    'Score': '得分',
}


def _build_result_table(work_dir: str) -> str:
    """Build a Markdown pipe-table from the JSON report files in *work_dir*/reports.

    Returns an empty string when no reports are found or on any error.
    """
    try:
        reports_dir = os.path.join(work_dir, 'reports')
        report_list = get_report_list([reports_dir])
        if not report_list:
            return ''
        df = get_data_frame(report_list, flatten_metrics=True, flatten_categories=True)
        _CAT_LEVEL_NAMES = ['类别', '子类别', '细分类别']
        new_cols = {}
        for col in df.columns:
            if col in _COLUMN_ZH:
                new_cols[col] = _COLUMN_ZH[col]
            elif col.startswith('Cat.'):
                try:
                    level = int(col[4:])
                    new_cols[col] = _CAT_LEVEL_NAMES[level] if level < len(_CAT_LEVEL_NAMES) else f'类别{level}'
                except ValueError:
                    new_cols[col] = col.replace('Cat.', '类别')
        df = df.rename(columns=new_cols)
        return tabulate(df, headers=df.columns, tablefmt='pipe', showindex=False)
    except Exception as e:
        logger.warning(f'Failed to build result table: {e}')
        return ''


_REQUIRED_FIELDS = ['model', 'datasets', 'api_url']


class RequestValidationError(Exception):
    """Raised by _parse_request when the incoming request is invalid."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@bp_eval.errorhandler(RequestValidationError)
def _handle_validation_error(exc: RequestValidationError):
    return jsonify({'error': exc.message}), exc.status_code


def _parse_request() -> tuple[dict, str]:
    """Validate the request body and return (data, task_id).

    Raises:
        RequestValidationError: when the request is missing required fields or
            the ``EvalScope-Task-Id`` header is absent or malformed.
    """
    data = request.get_json()
    if not data:
        raise RequestValidationError('Request body is required')

    for field in _REQUIRED_FIELDS:
        if field not in data:
            raise RequestValidationError(f'{field} is required')

    task_id = request.headers.get('EvalScope-Task-Id')
    if not task_id:
        raise RequestValidationError('EvalScope-Task-Id header is required')

    try:
        validate_task_id(task_id)
    except ValueError as e:
        raise RequestValidationError(str(e)) from e

    return data, task_id


def _build_task_config(data: dict) -> TaskConfig:
    """Build a TaskConfig from request data with common defaults applied."""
    if not data.get('eval_type'):
        data['eval_type'] = EvalType.OPENAI_API

    task_config = TaskConfig.from_dict(data)
    task_config.no_timestamp = True
    task_config.enable_progress_tracker = True
    task_config.analysis_report = True
    return task_config


def _execute_task(task_id: str, task_config: TaskConfig, label: str = 'Task'):
    """Run the evaluation subprocess and return a Flask response."""
    create_log_file(task_id, os.path.join('logs', 'eval_log.log'))
    try:
        result = run_in_subprocess(run_eval_wrapper, task_config)
        table_str = _build_result_table(task_config.work_dir)
        logger.info(f'[{task_id}] {label} completed successfully')
        return jsonify({'status': 'completed', 'task_id': task_id, 'result': result, 'table': table_str})
    except Exception as e:
        logger.error(f'[{task_id}] {label} failed: {e}')
        return jsonify({'status': 'error', 'task_id': task_id, 'error': str(e)}), 500


@bp_eval.route('/invoke', methods=['POST'])
def run_evaluation():
    """Run a model evaluation task (blocking).

    Returns the evaluation result when the task completes.
    """
    data, task_id = _parse_request()

    task_config = _build_task_config(data)
    task_config.work_dir = os.path.join(OUTPUT_DIR, task_id)

    logger.info(f'[{task_id}] Running evaluation task for model: {task_config.model}')
    logger.info(f'[{task_id}] Datasets: {task_config.datasets}')

    return _execute_task(task_id, task_config, label='Task')


@bp_eval.route('/resume/invoke', methods=['POST'])
def resume_evaluation():
    """Resume a previously interrupted evaluation task (blocking).

    Returns the evaluation result when the task completes.
    """
    data, task_id = _parse_request()

    work_dir = os.path.join(OUTPUT_DIR, task_id)
    if not os.path.isdir(work_dir):
        return jsonify({'error': f'Output directory not found for task_id: {task_id}'}), 404

    task_config = _build_task_config(data)
    task_config.work_dir = work_dir
    task_config.use_cache = work_dir
    task_config.rerun_review = True

    logger.info(f'[{task_id}] Running resume task, work_dir: {work_dir}')
    logger.info(f'[{task_id}] Model: {task_config.model}, Datasets: {task_config.datasets}')

    return _execute_task(task_id, task_config, label='Resume task')


@bp_eval.route('/progress', methods=['GET'])
def get_evaluation_progress():
    """Get the real-time hierarchical progress of a running evaluation task.

    Query params:
        task_id (str): the task identifier
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    progress_file = os.path.join(OUTPUT_DIR, task_id, 'progress.json')
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        return jsonify(progress), 200
    except FileNotFoundError:
        return jsonify({'percent': 0.0}), 200
    except Exception as e:
        logger.error(f'Failed to get progress for task {task_id}: {e}')
        return jsonify({'error': str(e)}), 500


@bp_eval.route('/report', methods=['GET'])
def get_evaluation_report():
    """Get the HTML evaluation report for a completed task.

    Query params:
        task_id (str): the task identifier
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    report_file = os.path.join(OUTPUT_DIR, task_id, 'reports', 'report.html')
    if not os.path.exists(report_file):
        return jsonify({'error': f'Report not found for task_id: {task_id}'}), 404

    return send_file(report_file, mimetype='text/html')


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

        try:
            content = get_log_content(task_id, os.path.join('logs', 'eval_log.log'), start_line)
        except FileNotFoundError:
            content = ''
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        logger.error(f'Failed to get evaluation log: {str(e)}')
        return jsonify({'error': str(e)}), 500


@bp_eval.route('/benchmarks', methods=['GET'])
def list_benchmarks():
    """Return the catalogue of supported benchmarks with descriptions.

    The list is split into two categories: ``text`` (LLM-only) and
    ``multimodal`` (VLM).  Descriptions are loaded from the ``_meta`` JSON
    files and post-processed: the H1 title and the last H2 section are
    stripped, then the remainder is split into per-section blocks.

    The default catalogue can be overridden at application startup by setting
    ``app.config['SUPPORTED_BENCHMARKS']`` to a dict with keys ``'text'`` and
    ``'multimodal'``, each containing a list of benchmark names.

    Query params:
        type (str, optional): Filter to ``'text'`` or ``'multimodal'`` only.
    """
    try:
        # Allow the catalogue to be overridden via Flask app config
        cfg = current_app.config.get('SUPPORTED_BENCHMARKS', {})
        text_names: List[str] = cfg.get('text', DEFAULT_TEXT_BENCHMARKS)
        multimodal_names: List[str] = cfg.get('multimodal', DEFAULT_MULTIMODAL_BENCHMARKS)

        filter_type = request.args.get('type', '').lower()

        result: Dict[str, Any] = {}
        if filter_type in ('', 'text'):
            result['text'] = [build_benchmark_entry(name) for name in text_names]
        if filter_type in ('', 'multimodal'):
            result['multimodal'] = [build_benchmark_entry(name) for name in multimodal_names]

        if filter_type and filter_type not in ('text', 'multimodal'):
            return jsonify({'error': f"Unknown type '{filter_type}'. Use 'text' or 'multimodal'."}), 400

        return jsonify(result), 200
    except Exception as e:
        logger.error(f'Failed to list benchmarks: {e}')
        return jsonify({'error': str(e)}), 500
