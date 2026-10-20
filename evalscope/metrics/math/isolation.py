# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded, process-isolated execution for Math-Verify calls.

Math-Verify protects its regex extraction with a ``signal.alarm`` timeout, which only
works on the main thread of the main interpreter. EvalScope scores samples inside a
thread pool, so :mod:`evalscope.metrics.math.parser` would otherwise have to call
Math-Verify with timeouts disabled. Degenerate model output then hangs the whole
evaluation: a prediction such as the same ``\\boxed{...}`` repeated thousands of times
makes the extraction regex backtrack catastrophically, and the match runs as C code
while holding the GIL, freezing every worker thread in the process.

Executing the call in a short-lived child process restores both guards: the call runs on
the child's main thread, so Math-Verify's native signal timeout applies again, and the
parent additionally enforces a hard kill if the child ignores even its own alarm.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from typing import Any, Callable, TypeVar

from evalscope.utils.logger import get_logger

logger = get_logger()

_T = TypeVar('_T')


class MathIsolationError(Exception):
    """A Math-Verify call crashed or exceeded its hard timeout in the worker process."""


def _isolated_child(conn, fn: Callable[..., Any], args: tuple) -> None:
    # Never touch inherited logging handlers: another parent thread could hold the logging
    # lock at fork time and the child would deadlock on its first log call.
    logging.disable(logging.CRITICAL)
    try:
        conn.send((True, fn(*args)))
    except BaseException as exc:  # noqa: BLE001 - the failure is reported to the parent
        try:
            # Truncate: some Math-Verify error messages embed the whole prediction text.
            conn.send((False, f'{type(exc).__name__}: {exc}'[:200]))
        except BaseException:  # noqa: BLE001 - nothing left to do if even reporting fails
            pass
    finally:
        conn.close()


def _start_with_retries(process: multiprocessing.Process, label: str, attempts: int = 5, delay: float = 0.2) -> None:
    """Start a worker, retrying transient fork guards.

    Concurrent users of ``filelock`` (dataset caches and friends) briefly block ``os.fork``
    while they shuffle descriptors; that window is short, so a bounded retry wins.
    """
    for attempt in range(attempts):
        try:
            process.start()
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def run_isolated(fn: Callable[..., _T], /, *args: Any, hard_timeout: float, label: str) -> _T:
    """Run ``fn(*args)`` in a child process and return its result.

    ``fn`` must be a top-level callable picklable by reference. A child crash or a call
    exceeding ``hard_timeout`` seconds raises :class:`MathIsolationError` after the child
    is terminated, so a hung regex can neither leak nor block the parent.
    """
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context('fork' if 'fork' in methods else 'spawn')
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(target=_isolated_child, args=(child_conn, fn, args), daemon=True)
    try:
        _start_with_retries(process, label)
    except Exception as exc:
        parent_conn.close()
        child_conn.close()
        raise MathIsolationError(f'{label}: could not start isolated worker: {exc}') from exc
    child_conn.close()
    try:
        if parent_conn.poll(hard_timeout):
            ok, payload = parent_conn.recv()
            if ok:
                return payload
            raise MathIsolationError(f'{label}: worker failed: {payload}')
        raise MathIsolationError(f'{label}: worker exceeded hard timeout of {hard_timeout}s')
    finally:
        parent_conn.close()
        if process.is_alive():
            process.terminate()
        process.join()
