# Copyright (c) Alibaba, Inc. and its affiliates.
"""Deadline-bounded execution of one mathematical scoring call in a forked child.

Math-Verify's worst case is unbounded: some degenerate predictions make its extraction regex
backtrack for minutes while holding the GIL, freezing every scoring thread. Only a process
boundary can preempt that, so each public call runs in a short-lived child forked from the
warm scoring process. Arguments and results stay plain data, and SymPy objects never cross the
boundary because pickling re-canonicalizes them.

The deadline covers computation and the complete result transfer once the child has started.
Warm-up, waiting for the fork lock, process start, decoding in the parent and cleanup are not
strictly bounded. Without fork, or inside a daemonic process, calls run in-process with no deadline.
"""

import logging
import multiprocessing
import pickle
import socket
import threading
import time
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


def _child(
    conn: socket.socket,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    deadline: float,
) -> None:
    import resource

    # Cap CPU time so a CPU-bound child orphaned by a killed parent eventually stops.
    # Never raise an inherited hard limit; this does not stop a child blocked while waiting.
    cpu_limit = int(deadline) + 5
    _, hard_limit = resource.getrlimit(resource.RLIMIT_CPU)
    if hard_limit != resource.RLIM_INFINITY:
        cpu_limit = min(cpu_limit, hard_limit)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    # Another parent thread may hold a logging lock at fork time.
    logging.disable(logging.CRITICAL)
    # Inherited filters keep the caller's policy, e.g. warnings raised as errors.
    with warnings.catch_warnings(record=True) as caught:
        try:
            payload = fn(*args, **kwargs)
            ok = True
        except BaseException as exc:
            payload = exc
            ok = False
    issued = [(str(item.message), item.category) for item in caught]
    try:
        try:
            data = pickle.dumps((ok, payload, issued))
        except Exception as exc:
            data = pickle.dumps(
                (False, MathEvaluationError(f'Unpicklable scoring outcome: {type(exc).__name__}: {exc}'), issued)
            )
        conn.sendall(data)
    finally:
        # End the single response explicitly: copies of this socket inherited by concurrently
        # forked children would otherwise delay EOF for the parent.
        conn.shutdown(socket.SHUT_WR)
        conn.close()


def _receive_result(receiver: socket.socket, expires_at: float) -> Any:
    """Read until EOF before ``expires_at``, then decode the complete response."""
    chunks = []
    while True:
        remaining = expires_at - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        receiver.settimeout(remaining)
        chunk = receiver.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    if not chunks:
        raise EOFError
    return pickle.loads(b''.join(chunks))


def run_isolated(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    deadline: float,
    label: str,
    *,
    kwargs: dict[str, Any] | None = None,
) -> Any:
    """Run ``fn(*args, **kwargs)`` in a forked child and return its result or re-raise its exception.

    Child warnings follow the caller's filters and are replayed in the caller. Computation and the
    complete response share one ``deadline`` that starts once the child is running. A child that
    cannot start, exits without a result or misses the deadline raises :class:`MathEvaluationError`.
    Lock waiting, process start, decoding in the parent and cleanup are not covered by the deadline.
    """
    context = multiprocessing.get_context('fork')
    receiver, sender = socket.socketpair()
    process = context.Process(target=_child, args=(sender, fn, args, kwargs or {}, deadline), daemon=True)
    try:
        try:
            with _FORK_LOCK:
                process.start()
        except Exception as exc:
            raise MathEvaluationError(f'{label}: could not start the scoring process: {exc}') from exc
        finally:
            sender.close()
        expires_at = time.monotonic() + deadline
        try:
            ok, payload, issued = _receive_result(receiver, expires_at)
        except TimeoutError as exc:
            raise MathEvaluationError(f'{label}: exceeded the {deadline}s mathematical scoring deadline') from exc
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
