"""Bounded reusable subprocesses with parent-enforced deadlines on every platform."""

import atexit
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .contracts import MathEvaluationError, MathRequest, MathResult


class _Worker:
    def __init__(self, command: list[str]) -> None:
        env = os.environ.copy()
        root = str(Path(__file__).resolve().parents[3])
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [root, env.get('PYTHONPATH', '')]))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding='utf-8',
            bufsize=1,
            env=env,
        )
        self.responses: queue.Queue[str | None] = queue.Queue()
        self.requests: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True, name='math-verify-reader')
        self.writer = threading.Thread(target=self._write, daemon=True, name='math-verify-writer')
        self.reader.start()
        self.writer.start()

    def _write(self) -> None:
        while (line := self.requests.get()) is not None:
            try:
                self.process.stdin.write(line)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                self.responses.put(None)
                return

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def execute(self, request: MathRequest, deadline: float) -> MathResult:
        try:
            # Pipe writes can also block (large requests or a stalled child). They
            # run outside the caller so the same deadline bounds both directions.
            self.requests.put(request.model_dump_json() + '\n')
            response = self.responses.get(timeout=max(0, deadline - time.monotonic()))
            if response is None:
                raise MathEvaluationError(f'Math-Verify worker exited ({self.process.poll()})')
            result = MathResult.model_validate_json(response)
            if result.error:
                raise MathEvaluationError(result.error)
            return result
        except queue.Empty as exc:
            raise MathEvaluationError('Math-Verify exceeded its total execution deadline') from exc
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise MathEvaluationError(f'Math-Verify worker communication failed: {exc}') from exc

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.requests.put(None)
        self.writer.join(timeout=1)
        self.reader.join(timeout=1)
        for stream in [self.process.stdin, self.process.stdout]:
            stream.close()


class MathWorkerPool:
    """Lease at most four workers; queueing and cold startup share the call deadline."""

    def __init__(
        self,
        max_workers: int = 4,
        timeout: float = 15,
        worker_command: list[str] | None = None,
    ) -> None:
        if not 1 <= max_workers <= 4 or timeout <= 0:
            raise ValueError('Use 1–4 Math-Verify workers and a positive deadline')
        self.timeout = timeout
        self.command = worker_command or [sys.executable, '-u', '-m', 'evalscope.metrics.math.worker']
        self._condition = threading.Condition()
        self._workers: dict[int, _Worker] = {}
        self._available = list(range(max_workers))
        self._closed = False

    def execute(self, request: MathRequest) -> MathResult:
        """Discard a worker after exceptions, crashes or cancellation."""
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while not self._available and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MathEvaluationError('Math-Verify worker queue exceeded its total deadline')
                self._condition.wait(remaining)
            if self._closed:
                raise MathEvaluationError('Math-Verify worker pool is closed')
            slot = self._available.pop()
            try:
                if slot not in self._workers:
                    self._workers[slot] = _Worker(self.command)
                worker = self._workers[slot]
            except BaseException:
                self._available.append(slot)
                self._condition.notify()
                raise
        try:
            return worker.execute(request, deadline)
        except BaseException:
            with self._condition:
                failed = self._workers.pop(slot, None)
            if failed is not None:
                failed.close()
            raise
        finally:
            with self._condition:
                self._available.append(slot)
                self._condition.notify()

    def close(self) -> None:
        """Terminate running and idle children and wake queued callers."""
        with self._condition:
            self._closed = True
            workers = list(self._workers.values())
            self._workers.clear()
            self._condition.notify_all()
        for worker in workers:
            worker.close()

    def __enter__(self) -> 'MathWorkerPool':
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


_lock = threading.RLock()
_pool: MathWorkerPool | None = None
_sessions = 0


def execute_math(request: MathRequest) -> MathResult:
    """Use the process-wide pool shared by evaluation threads."""
    global _pool
    with _lock:
        if _pool is None:
            _pool = MathWorkerPool()
        pool = _pool
    return pool.execute(request)


def shutdown_math_workers() -> None:
    """Reap all children before interpreter teardown."""
    global _pool
    with _lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.close()


@contextmanager
def math_worker_session() -> Iterator[None]:
    """Keep concurrent tasks' workers alive until the final task finishes."""
    global _sessions
    with _lock:
        _sessions += 1
    try:
        yield
    finally:
        with _lock:
            _sessions -= 1
            if _sessions == 0:
                shutdown_math_workers()


atexit.register(shutdown_math_workers)
