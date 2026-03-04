import multiprocessing
import threading
import traceback
from datetime import datetime

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Task store
# ---------------------------------------------------------------------------


class TaskStore:
    """Thread-safe in-memory store for tracking async task states.

    Each entry is a dict with at least the following keys:
        status      : 'running' | 'completed' | 'error'
        submitted_at: ISO-8601 timestamp
        finished_at : ISO-8601 timestamp (present when status != 'running')
        result      : task return value (present when status == 'completed')
        error       : error message string (present when status == 'error')
    """

    def __init__(self):
        self._tasks: dict = {}
        self._lock = threading.Lock()

    def set(self, task_id: str, data: dict):
        with self._lock:
            self._tasks[task_id] = data

    def update(self, task_id: str, data: dict):
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(data)
            else:
                self._tasks[task_id] = data

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            return dict(self._tasks[task_id]) if task_id in self._tasks else None

    def all(self) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self._tasks.items()}


# Singleton shared across the whole process
task_store = TaskStore()

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
# Async task submission
# ---------------------------------------------------------------------------


def submit_task(task_id: str, func, *args, **kwargs) -> str:
    """Submit *func* to run asynchronously in a background subprocess.

    Returns *task_id* immediately.  Poll ``task_store.get(task_id)`` for
    the current status / result.
    """
    task_store.set(task_id, {
        'status': 'running',
        'submitted_at': datetime.now().isoformat(),
    })

    def _run():
        try:
            result = run_in_subprocess(func, *args, **kwargs)
            task_store.update(
                task_id, {
                    'status': 'completed',
                    'result': result,
                    'finished_at': datetime.now().isoformat(),
                }
            )
            logger.info(f'[{task_id}] Task completed successfully')
        except Exception as e:
            task_store.update(
                task_id, {
                    'status': 'error',
                    'error': str(e),
                    'finished_at': datetime.now().isoformat(),
                }
            )
            logger.error(f'[{task_id}] Task failed: {e}')

    thread = threading.Thread(target=_run, daemon=True, name=f'task-{task_id}')
    thread.start()
    return task_id


# ---------------------------------------------------------------------------
# Task wrappers (thin shims kept for clarity / future extension)
# ---------------------------------------------------------------------------


def run_eval_wrapper(task_config: TaskConfig):
    """Run an evaluation task and return the result."""
    return run_task(task_config)


def run_perf_wrapper(perf_args: PerfArguments):
    """Run a performance benchmark and return the result."""
    return run_perf_benchmark(perf_args)
