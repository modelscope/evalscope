import contextlib
import io
import multiprocessing
import queue
import sys
import traceback

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _capture_stderr():
    """Context manager that redirects sys.stderr to a StringIO buffer.

    Yields the buffer so the caller can read captured output after the block.
    Always restores the original sys.stderr on exit.
    """
    buf = io.StringIO()
    original = sys.stderr
    sys.stderr = buf
    try:
        yield buf
    finally:
        sys.stderr = original


def _process_worker(func, result_queue, *args, **kwargs):
    """Target for multiprocessing.Process — executes *func* and posts result.

    stderr is captured and forwarded through the queue so the parent process
    can surface it even when the child crashes before *func* is reached.
    """
    with _capture_stderr() as stderr_buf:
        try:
            result = func(*args, **kwargs)
            result_queue.put({'status': 'success', 'result': result})
        except BaseException as e:
            result_queue.put({
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'stderr': stderr_buf.getvalue(),
            })


def run_in_subprocess(func, *args, **kwargs):
    """Run *func* in a child process and return its result (blocks caller).

    Returns the function's return value on success; raises on error.
    """
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_process_worker, args=(func, result_queue, *args), kwargs=kwargs)
    p.start()
    p.join()

    try:
        # Use a short timeout instead of queue.empty() to avoid race conditions
        # where the child has put data but the buffer hasn't flushed yet.
        res = result_queue.get(timeout=5)
        if res['status'] == 'error':
            stderr_info = res.get('stderr', '')
            stderr_section = f'\n[stderr]\n{stderr_info}' if stderr_info.strip() else ''
            raise RuntimeError(f"Subprocess error: {res['error']}\n{res.get('traceback', '')}{stderr_section}")
        return res['result']
    except queue.Empty:
        # Queue was truly empty — the child crashed before putting anything
        # (OOM, SIGKILL, import error, segfault, etc.).
        raise RuntimeError(
            f'Subprocess terminated unexpectedly (exit code {p.exitcode}). '
            'The child process may have crashed due to OOM, a missing import, '
            'GPU initialisation failure, or a signal (e.g. SIGKILL).'
        ) from None


# ---------------------------------------------------------------------------
# Task wrappers (thin shims kept for clarity / future extension)
# ---------------------------------------------------------------------------


def run_eval_wrapper(task_config: TaskConfig):
    """Run an evaluation task and return the result."""
    return run_task(task_config)


def run_perf_wrapper(perf_args: PerfArguments):
    """Run a performance benchmark and return the result."""
    return run_perf_benchmark(perf_args)
