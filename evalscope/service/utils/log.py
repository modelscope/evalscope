import os
from collections import deque

from evalscope.constants import DEFAULT_WORK_DIR

OUTPUT_DIR = os.path.abspath(os.getenv('EVALSCOPE_OUTPUT_DIR', DEFAULT_WORK_DIR))


def validate_task_id(task_id: str) -> None:
    """Validate a task_id value.

    Raises:
        ValueError: if task_id is empty or contains path-traversal characters.
    """
    if not task_id:
        raise ValueError('task_id is required')
    if os.path.basename(task_id) != task_id:
        raise ValueError('Invalid task_id')


def create_log_file(task_id: str, sub_path: str) -> str:
    """Create an empty log file for a given task so that log polling does not raise FileNotFoundError.

    Returns the absolute path of the created log file.
    """
    validate_task_id(task_id)

    log_file = os.path.join(OUTPUT_DIR, task_id, sub_path)
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

    log_file = os.path.join(OUTPUT_DIR, task_id, sub_path)
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
