import json
import os
import threading
import time
from contextlib import nullcontext
from typing import Optional, Union

from evalscope.utils import get_logger
from evalscope.utils.io_utils import current_time

logger = get_logger()


class ProgressTracker:
    """
    File-backed flat progress tracker for evaluation and perf pipelines.

    Tracks overall processed/total counts and writes a simple JSON snapshot
    to ``<work_dir>/progress.json`` on every update (throttled by
    ``write_interval``).  Status-change writes are always immediate.

    State schema written to ``<work_dir>/progress.json``::

        {
          "status":          "running" | "completed" | "error",
          "pipeline":        "eval" | "perf",
          "total_count":     14042,
          "processed_count": 5200,
          "percent":         37.03,
          "updated_at":      "2026-04-02T10:05:42"
        }

    ``write_interval`` controls how often incremental progress updates are
    flushed to disk (in seconds).

    Usage::

        with make_tracker(enabled, work_dir, pipeline='eval', total_count=5000):
            # inside the pipeline call ProgressTracker.get_current().update(n)
            ...
    """

    _current: Optional['ProgressTracker'] = None

    def __init__(
        self,
        work_dir: str,
        pipeline: str = '',
        write_interval: float = 1.0,
        total_count: Optional[int] = None,
    ):
        os.makedirs(work_dir, exist_ok=True)
        self._path = os.path.join(work_dir, 'progress.json')
        self._lock = threading.Lock()
        self._processed_count: int = 0
        self._status = 'running'
        self._pipeline = pipeline
        self._write_interval = write_interval
        self._last_write_time: float = 0.0
        if total_count is None or total_count <= 0:
            raise ValueError(f'`total_count` must be > 0, got {total_count}.')
        self._total_count: int = total_count
        self._write(force=True)

    # ------------------------------------------------------------------
    # Context manager — wraps the full pipeline lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> 'ProgressTracker':
        ProgressTracker._current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.set_status('error')
        else:
            self.set_status('completed')
        ProgressTracker._current = None
        return False

    # ------------------------------------------------------------------
    # Class-level accessor
    # ------------------------------------------------------------------

    @classmethod
    def get_current(cls) -> Optional['ProgressTracker']:
        """Return the currently active ProgressTracker, or None."""
        return cls._current

    # ------------------------------------------------------------------
    # Progress API
    # ------------------------------------------------------------------

    def update(self, n: int = 1) -> None:
        """Increment processed_count by *n* and flush to disk (throttled)."""
        with self._lock:
            self._processed_count += n
            self._write(force=False)

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status
            self._write(force=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, force: bool = True) -> None:
        """Write progress state to disk.

        When ``force=False`` the write is skipped if less than ``write_interval``
        seconds have elapsed since the last write.  The lock must already be held
        by the caller.
        """
        now = time.monotonic()
        if not force and (now - self._last_write_time) < self._write_interval:
            return
        self._last_write_time = now
        processed = self._processed_count
        percent: Optional[float] = None
        if self._total_count > 0:
            percent = round(processed / self._total_count * 100, 2)
        state = {
            'status': self._status,
            'pipeline': self._pipeline,
            'total_count': self._total_count,
            'processed_count': processed,
            'percent': percent,
            'updated_at': current_time().isoformat(),
        }
        logger.debug(f'Processed / Total: {processed} / {self._total_count} | {percent}%')
        tmp = self._path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._path)


def make_tracker(
    enabled: bool,
    work_dir: str,
    pipeline: str = '',
    write_interval: float = 1.0,
    total_count: Optional[int] = None,
) -> Union['ProgressTracker', 'nullcontext']:
    """Return a ``ProgressTracker`` context manager when *enabled*, otherwise a no-op ``nullcontext``.

    Example::

        with make_tracker(task_config.enable_progress_tracker, outputs.outputs_dir, pipeline='eval', total_count=5000):
            ...
    """
    if enabled and total_count is not None and total_count > 0:
        return ProgressTracker(
            work_dir=work_dir, pipeline=pipeline, write_interval=write_interval, total_count=total_count
        )
    return nullcontext()
