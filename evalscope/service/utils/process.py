import multiprocessing
import os
import traceback
import uuid
from datetime import datetime
from flask import jsonify, request
from functools import wraps

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.run import run_task
from evalscope.utils.logger import get_logger
from .log import OUTPUT_DIR, TASK_FINISH_MARKER, TASK_START_MARKER, LogManager

logger = get_logger()


def run_eval_wrapper(task_config: TaskConfig):
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
