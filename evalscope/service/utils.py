import itertools
import multiprocessing
import os
import traceback
import uuid
from datetime import datetime
from flask import jsonify, request
from functools import wraps

from evalscope.config import TaskConfig
from evalscope.constants import DEFAULT_WORK_DIR
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.run import run_task
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


def run_task_wrapper(task_config: TaskConfig):
    """Wrapper to run evaluation task with log markers."""
    log_file = os.path.join(task_config.work_dir, 'logs', 'eval_log.log')
    LogManager.append(log_file, TASK_START_MARKER.format(datetime.now().isoformat()))
    try:
        return run_task(task_config)
    finally:
        LogManager.append(log_file, TASK_FINISH_MARKER.format(datetime.now().isoformat()))


def run_perf_wrapper(perf_args: PerfArguments):
    """Wrapper to run performance test with log markers."""
    log_file = os.path.join(perf_args.outputs_dir, 'perf', 'benchmark.log')
    LogManager.append(log_file, TASK_START_MARKER.format(datetime.now().isoformat()))
    try:
        return run_perf_benchmark(perf_args)
    finally:
        LogManager.append(log_file, TASK_FINISH_MARKER.format(datetime.now().isoformat()))


def _process_worker(func, queue, *args, **kwargs):
    """Worker function to run task in a separate process."""
    try:
        result = func(*args, **kwargs)
        queue.put({'status': 'success', 'result': result})
    except Exception as e:
        queue.put({'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()})


def run_in_subprocess(func, *args, **kwargs):
    """Run a function in a subprocess and return the result."""
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_process_worker, args=(func, queue, *args), kwargs=kwargs)
    p.start()
    p.join()

    if not queue.empty():
        res = queue.get()
        if res['status'] == 'error':
            raise Exception(f"Subprocess error: {res['error']}\n{res.get('traceback', '')}")
        return res['result']
    else:
        raise Exception(f'Subprocess terminated unexpectedly with exit code {p.exitcode}')


def handle_exceptions(log_subpath: str = 'error.log'):
    """Decorator to handle exceptions in route handlers."""

    def decorator(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            request_id = request.headers.get('X-Fc-Request-Id', uuid.uuid4().hex)

            try:
                return f(*args, request_id=request_id, **kwargs)
            except Exception as e:
                logger.error(f'Request failed: {str(e)}')
                logger.error(traceback.format_exc())

                # Write error to log file
                work_dir = os.path.join(OUTPUT_DIR, request_id)
                LogManager.log_error(work_dir, log_subpath, traceback.format_exc())
                LogManager.log_error(work_dir, log_subpath, TASK_FINISH_MARKER.format(datetime.now().isoformat()))

                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'request_id': request_id
                }), 500

        return wrapper

    return decorator


def get_log_content(request_id: str, sub_path: str, start_line: int = 0):
    """Helper to read log content."""
    if not request_id:
        raise ValueError('request_id is required')

    log_file = os.path.join(OUTPUT_DIR, request_id, sub_path)
    if not os.path.exists(log_file):
        raise FileNotFoundError(f'Log file not found: {log_file}')

    with open(log_file, 'r', encoding='utf-8') as f:
        if start_line > 0:
            content = ''.join(itertools.islice(f, start_line, None))
        else:
            content = f.read()
    return content
