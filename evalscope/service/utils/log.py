from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

from evalscope.constants import DEFAULT_WORK_DIR
from evalscope.service.api_models.common import ActiveTaskInfo, TaskConfigSummary
from evalscope.utils.uri_utils import sanitize_endpoint

if TYPE_CHECKING:
    from evalscope.config import TaskConfig
    from evalscope.perf.arguments import Arguments as PerfArguments

OUTPUT_DIR = os.path.abspath(os.getenv('EVALSCOPE_OUTPUT_DIR', DEFAULT_WORK_DIR))

TASK_INFO_FILENAME = 'task_info.json'


def validate_task_id(task_id: str) -> None:
    """Validate a task_id value.

    Raises:
        ValueError: if task_id is empty or contains path-traversal characters.
    """
    if not isinstance(task_id, str) or not task_id:
        raise ValueError('task_id is required')
    if task_id in ('.', '..') or os.path.basename(task_id) != task_id or '\\' in task_id or '\x00' in task_id:
        raise ValueError('Invalid task_id')
    task_path = os.path.realpath(os.path.join(OUTPUT_DIR, task_id))
    root = os.path.realpath(OUTPUT_DIR)
    if os.path.commonpath([root, task_path]) != root or task_path == root:
        raise ValueError('Invalid task_id')


def task_file_path(task_id: str, sub_path: str) -> str:
    validate_task_id(task_id)
    task_dir = os.path.realpath(os.path.join(OUTPUT_DIR, task_id))
    path = os.path.realpath(os.path.join(task_dir, sub_path))
    if os.path.commonpath([task_dir, path]) != task_dir or path == task_dir:
        raise ValueError('Invalid task path')
    return path


def create_log_file(task_id: str, sub_path: str) -> str:
    """Create an empty log file for a given task so that log polling does not raise FileNotFoundError.

    Returns the absolute path of the created log file.
    """
    validate_task_id(task_id)

    log_file = task_file_path(task_id, sub_path)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8'):
            pass
    return log_file


def get_log_content(task_id: str, sub_path: str, start_line: int = None, page: int = 500) -> dict:
    """Read log content for a given task with pagination support.

    Args:
        task_id: The task identifier.
        sub_path: The log file path relative to task output directory.
        start_line: If None, read last `page` lines from end; otherwise read from this line (must be >= 0).
        page: Number of lines to read (must be >= 1, default 500).

    Returns:
        dict with keys:
            - text: log content (lines kept as-is, preserving original newlines)
            - head_line: 0-based start line number of returned content
            - tail_line: 0-based end line number (exclusive)
            - total_lines: total line count of the log file
    """
    validate_task_id(task_id)

    # Validate parameters
    if page < 1:
        raise ValueError('page must be >= 1')
    if start_line is not None and start_line < 0:
        raise ValueError('start_line must be >= 0')

    log_file = task_file_path(task_id, sub_path)
    if not os.path.exists(log_file):
        return {'text': '', 'head_line': 0, 'tail_line': 0, 'total_lines': 0}

    with open(log_file, 'r', encoding='utf-8') as f:
        # Single-pass: count lines and collect requested lines
        # This ensures total_lines matches Python's line iteration semantics
        total_lines = 0
        lines = deque(maxlen=page) if start_line is None else []

        for line in f:
            total_lines += 1
            if start_line is None:
                # deque with maxlen automatically keeps last N lines (O(1))
                lines.append(line)
            else:
                if total_lines > start_line and len(lines) < page:
                    lines.append(line)

        # Compute head_line based on actual total_lines
        if start_line is None:
            head_line = max(0, total_lines - page)
            lines = list(lines)
        elif start_line >= total_lines:
            return {'text': '', 'head_line': total_lines, 'tail_line': total_lines, 'total_lines': total_lines}
        else:
            head_line = start_line

    tail_line = head_line + len(lines)
    # Use ''.join to preserve original newlines in each line
    return {'text': ''.join(lines), 'head_line': head_line, 'tail_line': tail_line, 'total_lines': total_lines}


def write_task_manifest(task_id: str, manifest: dict) -> str:
    """Persist a task's display identity (alias / endpoint / start time) beside its outputs.

    Written once at task launch so that historical report listings can attribute
    a run to its endpoint long after the in-memory task registry is gone.
    Returns the absolute path of the manifest file.
    """
    validate_task_id(task_id)

    manifest_file = task_file_path(task_id, TASK_INFO_FILENAME)
    os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=os.path.dirname(manifest_file), delete=False) as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        temporary_path = f.name
    os.replace(temporary_path, manifest_file)
    return manifest_file


def build_task_info(
    task_id: str, kind: Literal['eval', 'perf'], config: TaskConfig | PerfArguments, alias: Optional[str]
) -> ActiveTaskInfo:
    if kind == 'eval':
        fields = ('datasets', 'limit', 'eval_batch_size', 'repeats')
        endpoint_url = getattr(config, 'api_url', None)
    else:
        fields = (
            'api',
            'dataset',
            'parallel',
            'number',
            'rate',
            'max_tokens',
            'min_tokens',
            'max_prompt_length',
            'min_prompt_length',
        )
        endpoint_url = getattr(config, 'url', None)

    summary = {}
    for key in fields:
        value = getattr(config, key, None)
        if value is not None:
            summary[key] = value

    if kind == 'eval':
        generation = getattr(config, 'generation_config', None)
        for key in ('temperature', 'top_p', 'top_k', 'max_tokens'):
            if isinstance(generation, dict):
                value = generation.get(key)
            else:
                value = getattr(generation, key, None)
            if value is not None:
                summary[key] = value
    info = ActiveTaskInfo(
        task_id=task_id,
        kind=kind,
        model=str(getattr(config, 'model', '') or ''),
        alias=alias,
        endpoint=sanitize_endpoint(endpoint_url),
        started_at=datetime.now().isoformat(),
        status='preparing',
        config=TaskConfigSummary(**summary),
    )
    write_task_manifest(task_id, info.model_dump(mode='json'))
    return info


def finish_task(task_id: str, status: Literal['completed', 'stopped', 'error']) -> None:
    manifest = read_task_manifest(os.path.join(OUTPUT_DIR, task_id))
    if manifest:
        manifest['status'] = status
        write_task_manifest(task_id, manifest)


def read_task_manifest(run_dir: str) -> Optional[dict]:
    """Read a task manifest from a run directory; ``None`` when absent or unreadable."""
    manifest_file = os.path.join(run_dir, TASK_INFO_FILENAME)
    if os.path.commonpath([os.path.realpath(run_dir), os.path.realpath(manifest_file)]) != os.path.realpath(run_dir):
        return None
    try:
        with open(manifest_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None
