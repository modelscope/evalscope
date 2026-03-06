import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional


class ProgressTracker:
    """
    File-backed hierarchical progress tracker for evaluation and perf pipelines.

    Each nesting level of TqdmLogging contributes one 'stage' entry.  Completed
    stages are retained in the JSON so callers can see full history.

    State schema written to ``<work_dir>/progress.json``::

        {
          "status":   "running" | "completed" | "error",
          "pipeline": "eval" | "perf",
          "stages": [
            {"depth": 0, "name": "Evaluating", "label": "mmlu",
             "current": 1, "total": 3, "status": "running"},
            {"depth": 1, "name": "Predicting", "label": "mmlu@test",
             "current": 1000, "total": 1000, "status": "completed"},
            {"depth": 1, "name": "Reviewing",  "label": "mmlu@test",
             "current": 320,  "total": 1000, "status": "running"}
          ],
          "updated_at": "2026-03-05T10:05:42Z"
        }

    ``depth`` reflects nesting level; list order is chronological.

    ``write_interval`` controls how often incremental progress updates are
    flushed to disk (in seconds).  Structural events (stage enter/exit, status
    change) are always written immediately regardless of the interval.
    """

    def __init__(self, work_dir: str, pipeline: str = '', write_interval: float = 1.0):
        os.makedirs(work_dir, exist_ok=True)
        self._path = os.path.join(work_dir, 'progress.json')
        self._lock = threading.Lock()
        self._stages: List[dict] = []
        # Maps nesting depth -> index in self._stages for currently *active* stages.
        self._active_depths: Dict[int, int] = {}
        self._status = 'running'
        self._pipeline = pipeline
        self._write_interval = write_interval
        self._last_write_time: float = 0.0
        self._write(force=True)

    # ------------------------------------------------------------------
    # Context manager — wraps the full pipeline lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> 'ProgressTracker':
        """Attach this tracker to TqdmLogging and return self."""
        from evalscope.utils.tqdm_utils import TqdmLogging
        TqdmLogging.set_tracker(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Detach tracker and finalise status based on whether an exception occurred."""
        from evalscope.utils.tqdm_utils import TqdmLogging
        self.set_status('error' if exc_type is not None else 'completed')
        TqdmLogging.set_tracker(None)
        return False  # never suppress exceptions

    # ------------------------------------------------------------------
    # Status API
    # ------------------------------------------------------------------

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status
            self._write(force=True)

    def _enter_stage(self, name: str, total: int, label: str) -> int:
        """Push a new stage onto the stack; returns the nesting depth."""
        with self._lock:
            depth = len(self._active_depths)
            idx = len(self._stages)
            self._stages.append({
                'depth': depth,
                'name': name,
                'label': label,
                'current': 0,
                'total': total,
                'status': 'running',
            })
            self._active_depths[depth] = idx
            self._write(force=True)
        return depth

    def _update_stage(
        self,
        depth: int,
        current: int,
        name: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None:
        """Update progress (and optionally name/label) for the active stage at *depth*.

        Disk writes are throttled to ``write_interval`` seconds to avoid I/O
        overhead on hot paths (e.g. perf benchmark inner loops).
        """
        with self._lock:
            if depth in self._active_depths:
                idx = self._active_depths[depth]
                self._stages[idx]['current'] = current
                if name is not None:
                    self._stages[idx]['name'] = name
                if label is not None:
                    self._stages[idx]['label'] = label
                self._write(force=False)

    def _exit_stage(self, depth: int) -> None:
        """Mark the stage at *depth* (and any orphaned child stages) as completed."""
        with self._lock:
            for d in [d for d in self._active_depths if d >= depth]:
                idx = self._active_depths.pop(d)
                self._stages[idx]['status'] = 'completed'
                self._stages[idx]['current'] = self._stages[idx]['total']
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
        state = {
            'status': self._status,
            'pipeline': self._pipeline,
            'stages': list(self._stages),
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        tmp = self._path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._path)
