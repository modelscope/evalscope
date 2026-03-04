import itertools
import os

from evalscope.constants import DEFAULT_WORK_DIR

OUTPUT_DIR = os.getenv('EVALSCOPE_OUTPUT_DIR', DEFAULT_WORK_DIR)


def create_log_file(task_id: str, sub_path: str) -> str:
    """Create an empty log file for a given task so that log polling does not raise FileNotFoundError.

    Returns the absolute path of the created log file.
    """
    if not task_id:
        raise ValueError('task_id is required')

    if not os.path.basename(task_id) == task_id:
        raise ValueError('Invalid task_id')

    log_file = os.path.join(OUTPUT_DIR, task_id, sub_path)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if not os.path.exists(log_file):
        open(log_file, 'w', encoding='utf-8').close()
    return log_file


def get_log_content(task_id: str, sub_path: str, start_line: int = 0) -> str:
    """Read log content for a given task."""
    if not task_id:
        raise ValueError('task_id is required')

    # Prevent path traversal
    if not os.path.basename(task_id) == task_id:
        raise ValueError('Invalid task_id')

    log_file = os.path.join(OUTPUT_DIR, task_id, sub_path)
    if not os.path.exists(log_file):
        raise FileNotFoundError(f'Log file not found: {log_file}')

    with open(log_file, 'r', encoding='utf-8') as f:
        if start_line > 0:
            content = ''.join(itertools.islice(f, start_line, None))
        else:
            content = f.read()
    return content
