import itertools
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
        start_line: If None, read last `page` lines from end; otherwise read from this line.
        page: Number of lines to read (default 500).

    Returns:
        dict with keys:
            - text: log content, lines joined by '\n'
            - head_line: 0-based start line number of returned content
            - tail_line: 0-based end line number (exclusive)
            - total_lines: total line count of the log file
    """
    validate_task_id(task_id)

    log_file = os.path.join(OUTPUT_DIR, task_id, sub_path)
    if not os.path.exists(log_file):
        return {'text': '', 'head_line': 0, 'tail_line': 0, 'total_lines': 0}

    with open(log_file, 'r', encoding='utf-8') as f:
        # Fast line count: count newlines in chunks
        total_lines = 0
        while True:
            chunk = f.read(65536)  # 64KB chunks
            if not chunk:
                break
            total_lines += chunk.count('\n')

        f.seek(0)

        if start_line is None:
            # Read last `page` lines from end
            lines = list(deque(f, maxlen=page))
            head_line = max(0, total_lines - page)
        else:
            # Read from start_line
            if start_line >= total_lines:
                return {'text': '', 'head_line': start_line, 'tail_line': total_lines, 'total_lines': total_lines}
            lines = list(itertools.islice(f, start_line, start_line + page))
            head_line = start_line

    tail_line = head_line + len(lines)
    return {'text': '\n'.join(lines), 'head_line': head_line, 'tail_line': tail_line, 'total_lines': total_lines}
