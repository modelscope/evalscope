import multiprocessing
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


def _process_worker(func, queue, *args, **kwargs):
    """Target for multiprocessing.Process — executes *func* and posts result."""
    try:
        result = func(*args, **kwargs)
        queue.put({'status': 'success', 'result': result})
    except Exception as e:
        queue.put({'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()})


def run_in_subprocess(func, *args, **kwargs):
    """Run *func* in a child process and return its result (blocks caller).

    Returns the function's return value on success; raises on error.
    """
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_process_worker, args=(func, queue, *args), kwargs=kwargs)
    p.start()
    p.join()

    if not queue.empty():
        res = queue.get()
        if res['status'] == 'error':
            raise RuntimeError(f"Subprocess error: {res['error']}\n{res.get('traceback', '')}")
        return res['result']
    else:
        raise RuntimeError(f'Subprocess terminated unexpectedly (exit code {p.exitcode})')


# ---------------------------------------------------------------------------
# Task wrappers (thin shims kept for clarity / future extension)
# ---------------------------------------------------------------------------


def run_eval_wrapper(task_config: TaskConfig):
    """Run an evaluation task and return the result."""
    return run_task(task_config)


def run_perf_wrapper(perf_args: PerfArguments):
    """Run a performance benchmark and return the result."""
    return run_perf_benchmark(perf_args)
