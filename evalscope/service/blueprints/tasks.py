import json
import os

from flask import Blueprint, jsonify

from evalscope.service.api_models import ListActiveTasksResponse
from evalscope.utils.logger import get_logger

from ..responses import json_response
from ..utils import OUTPUT_DIR, list_active_tasks

logger = get_logger()

bp_tasks = Blueprint('tasks', __name__, url_prefix='/api/v1/tasks')

# Progress tracker file location relative to the task output directory, per kind.
_PROGRESS_SUBPATH = {
    'eval': 'progress.json',
    'perf': os.path.join('perf', 'progress.json'),
}


def _read_progress_percent(task_id: str, kind: str) -> float:
    """Best-effort progress percentage from the task's tracker file; 0 when absent."""
    sub_path = _PROGRESS_SUBPATH.get(kind)
    if not sub_path:
        return 0.0
    progress_file = os.path.join(OUTPUT_DIR, task_id, sub_path)
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
    except (OSError, ValueError):
        return 0.0
    percent = progress.get('percent')
    return float(percent) if isinstance(percent, (int, float)) else 0.0


@bp_tasks.route('/active', methods=['GET'])
def list_active():
    """List tasks whose subprocess is currently registered on this service.

    The registry is in-memory: tasks vanish from this list when they finish or
    when the service restarts (a restart kills its child processes anyway).
    """
    try:
        tasks = []
        for record in list_active_tasks():
            record['percent'] = _read_progress_percent(record['task_id'], record.get('kind', ''))
            tasks.append(record)
        tasks.sort(key=lambda t: t.get('started_at', ''), reverse=True)
        return json_response(ListActiveTasksResponse, {'tasks': tasks, 'total': len(tasks)})
    except Exception as e:
        logger.error(f'Failed to list active tasks: {e}')
        return jsonify({'error': str(e)}), 500
