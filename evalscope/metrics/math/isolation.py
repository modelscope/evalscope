# Copyright (c) Alibaba, Inc. and its affiliates.
"""Deadline-bounded execution of one mathematical scoring call in a forked child.

Math-Verify's worst case is unbounded: some degenerate predictions make its extraction regex
backtrack for minutes while holding the GIL, freezing every scoring thread. Only a process
boundary can preempt that, so each public call runs in a short-lived child forked from the
warm scoring process. Arguments and results stay plain data, and SymPy objects never cross the
boundary because pickling re-canonicalizes them. A call that misses its deadline is killed and
reported as an execution failure.
"""

import logging
import multiprocessing
import threading
import warnings
from typing import Any, Callable

from .contracts import MathEvaluationError

_FORK_AVAILABLE = 'fork' in multiprocessing.get_all_start_methods()
# Serialize forks: at-fork handlers such as filelock's reject os.fork while another thread is forking.
_FORK_LOCK = threading.Lock()


def isolation_available() -> bool:
    """Report whether calls can be isolated; otherwise they run in-process without a deadline."""
    # Daemonic processes may not have children.
    return _FORK_AVAILABLE and not multiprocessing.current_process().daemon


def _child(conn: Any, fn: Callable[..., Any], args: tuple, deadline: float) -> None:
    import resource

    # Self-terminate even if the parent dies before it can kill this child.
    cpu_limit = int(deadline) + 5
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    # Another parent thread may hold a logging lock at fork time.
    logging.disable(logging.CRITICAL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            ok, payload = True, fn(*args)
        except BaseException as exc:
            ok, payload = False, exc
    issued = [(str(item.message), item.category) for item in caught]
    try:
        conn.send((ok, payload, issued))
    except Exception as exc:
        conn.send((False, MathEvaluationError(f'Unpicklable scoring outcome: {type(exc).__name__}: {exc}'), issued))
    finally:
        conn.close()


def run_isolated(fn: Callable[..., Any], args: tuple, deadline: float, label: str) -> Any:
    """Return ``fn(*args)`` computed in a forked child, re-raising its exception unchanged.

    Warnings issued by the child are replayed in the caller. A child that cannot start, exits
    without a result or exceeds ``deadline`` seconds raises :class:`MathEvaluationError`.
    """
    context = multiprocessing.get_context('fork')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(sender, fn, args, deadline), daemon=True)
    try:
        try:
            with _FORK_LOCK:
                process.start()
        except Exception as exc:
            raise MathEvaluationError(f'{label}: could not start the scoring process: {exc}') from exc
        finally:
            sender.close()
        if not receiver.poll(deadline):
            raise MathEvaluationError(f'{label}: exceeded the {deadline}s mathematical scoring deadline')
        try:
            ok, payload, issued = receiver.recv()
        except EOFError as exc:
            raise MathEvaluationError(f'{label}: scoring process exited with code {process.exitcode}') from exc
    finally:
        receiver.close()
        if process.pid is not None:
            process.kill()
            process.join()
    for message, category in issued:
        warnings.warn(message, category, stacklevel=3)
    if not ok:
        raise payload
    return payload
