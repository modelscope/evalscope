import itertools
import os
from datetime import datetime

from evalscope.constants import DEFAULT_WORK_DIR
from evalscope.utils.logger import get_logger

logger = get_logger()

OUTPUT_DIR = os.getenv('EVALSCOPE_OUTPUT_DIR', DEFAULT_WORK_DIR)
TASK_START_MARKER = '*** [EvalScope Service] Task started at {} ***'
TASK_FINISH_MARKER = '*** [EvalScope Service] Task finished at {} ***'


class LogManager:
    """Helper class to manage log files."""

    @staticmethod
    def get_log_path(work_dir: str, sub_path: str) -> str:
        return os.path.join(work_dir, sub_path)

    @staticmethod
    def append(file_path: str, content: str):
        """Append content to log file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f'{content}\n')
        except Exception as e:
            logger.error(f'Failed to write to log {file_path}: {e}')

    @staticmethod
    def log_error(work_dir: str, sub_path: str, error_msg: str):
        """Write error message with timestamp to log file."""
        log_file = LogManager.get_log_path(work_dir, sub_path)
        content = f'\n[Error] {datetime.now().isoformat()}\n{error_msg}'
        LogManager.append(log_file, content)


def get_log_content(task_id: str, sub_path: str, start_line: int = 0):
    """Helper to read log content."""
    if not task_id:
        raise ValueError('task_id is required')

    # Ensure task_id is a valid, not .. / etc
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
