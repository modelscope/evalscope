import json
import math
import os
from urllib.parse import quote, urlencode

from flask import Blueprint, jsonify
from flask.typing import ResponseReturnValue
from pydantic import ValidationError

from evalscope.report.data_frames import scan_report_refs
from evalscope.service.api_models import ActiveTaskInfo, ListActiveTasksResponse
from evalscope.service.api_models.common import TaskConfigSummary
from evalscope.utils.logger import get_logger
from evalscope.utils.uri_utils import sanitize_endpoint

from ..responses import json_response
from ..utils import OUTPUT_DIR, list_active_tasks, read_task_manifest, task_file_path, validate_task_id

logger = get_logger()

bp_tasks = Blueprint('tasks', __name__, url_prefix='/api/v1/tasks')

# Progress tracker file location relative to the task output directory, per kind.
_PROGRESS_SUBPATH = {
    'eval': 'progress.json',
    'perf': os.path.join('perf', 'progress.json'),
}


def _read_progress_percent(task_id: str, kind: str) -> float | None:
    sub_path = _PROGRESS_SUBPATH.get(kind)
    if not sub_path:
        return None
    try:
        progress_file = task_file_path(task_id, sub_path)
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(progress, dict):
        return None
    percent = progress.get('percent')
    if type(percent) not in (int, float) or not math.isfinite(percent):
        return None
    return max(0.0, min(100.0, float(percent)))


def _with_progress(record: dict) -> dict:
    percent = _read_progress_percent(record['task_id'], record.get('kind', ''))
    record['percent'] = percent
    if percent is not None and percent >= 100:
        record['status'] = 'finalizing'
    else:
        record['status'] = 'running'
    return record


def _saved_task_info(task_id: str, manifest: dict) -> ActiveTaskInfo:
    values = {key: manifest[key] for key in ('kind', 'model', 'alias', 'started_at') if key in manifest}
    config = manifest.get('config')
    if isinstance(config, dict):
        values['config'] = {key: value for key, value in config.items() if key in TaskConfigSummary.model_fields}
    elif config is not None:
        values['config'] = config
    status = manifest.get('status')
    if status not in ('completed', 'error', 'stopped'):
        status = 'unknown'
    return ActiveTaskInfo(
        **values,
        task_id=task_id,
        endpoint=sanitize_endpoint(manifest.get('endpoint') or manifest.get('api_url')),
        status=status,
    )


def _set_report_links(info: ActiveTaskInfo, run_dir: str) -> None:
    if info.status not in ('completed', 'unknown'):
        return
    task_id = info.task_id
    root_query = urlencode({'root_path': OUTPUT_DIR})
    task_query = urlencode({'task_id': task_id})
    if info.kind == 'perf':
        from ..perf_archive import build_run_summary

        perf_dir = os.path.join(run_dir, 'perf')
        if os.path.isfile(os.path.join(perf_dir, 'perf_report.html')):
            info.html_report_url = f'/api/v1/perf/report?{task_query}'
        summary = build_run_summary(task_id + '/perf', perf_dir)
        if summary and summary['num_runs'] > 0:
            path_query = urlencode({'path': task_id + '/perf'})
            info.report_url = f'/perf-report?{root_query}&{path_query}'
        return

    refs = [ref for ref in scan_report_refs(OUTPUT_DIR) if ref.run_id == task_id]
    if len(refs) == 1:
        run_id = quote(task_id, safe='')
        model_id = quote(refs[0].model_id, safe='')
        info.report_url = f'/reports/{run_id}/{model_id}?{root_query}'
    elif refs:
        search_query = urlencode({'search': task_id})
        info.report_url = f'/reports?{root_query}&{search_query}'
    if os.path.isfile(os.path.join(run_dir, 'reports', 'report.html')):
        info.html_report_url = f'/api/v1/eval/report?{task_query}'


@bp_tasks.route('/<task_id>', methods=['GET'])
def get_task(task_id: str) -> ResponseReturnValue:
    try:
        validate_task_id(task_id)
        task_file_path(task_id, 'perf')
        task_file_path(task_id, 'reports')
    except ValueError:
        return jsonify({'error': 'Invalid task_id'}), 400
    active = next((item for item in list_active_tasks() if item['task_id'] == task_id), None)
    if active:
        return json_response(ActiveTaskInfo, _with_progress(active))
    run_dir = os.path.join(OUTPUT_DIR, task_id)
    manifest = read_task_manifest(run_dir)
    if manifest is None:
        if not os.path.isdir(run_dir):
            return jsonify({'error': 'Task not found'}), 404
        manifest = {'kind': 'perf' if os.path.isdir(os.path.join(run_dir, 'perf')) else 'eval'}
    try:
        info = _saved_task_info(task_id, manifest)
    except ValidationError:
        return jsonify({'error': 'Task metadata is unreadable'}), 400
    info.percent = _read_progress_percent(task_id, info.kind)
    _set_report_links(info, run_dir)
    return json_response(ActiveTaskInfo, info.model_dump(mode='json'))


@bp_tasks.route('/active', methods=['GET'])
def list_active() -> ResponseReturnValue:
    """List tasks whose subprocess is currently registered on this service.

    The registry is in-memory: tasks vanish from this list when they finish or
    when the service restarts. This registry does not recover or supervise
    child processes left behind by an abrupt service shutdown.
    """
    try:
        tasks = []
        for record in list_active_tasks():
            tasks.append(_with_progress(record))
        tasks.sort(key=lambda t: t.get('started_at', ''), reverse=True)
        return json_response(ListActiveTasksResponse, {'tasks': tasks, 'total': len(tasks)})
    except Exception as e:
        logger.error(f'Failed to list active tasks: {e}')
        return jsonify({'error': str(e)}), 500
