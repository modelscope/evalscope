import asyncio
import pytest
import threading
import time
from typing import AsyncGenerator

from evalscope.utils.function_utils import AsyncioLoopRunner, AsyncioLoopThread


async def _current_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


def test_asyncio_loop_thread_survives_caller_loop_shutdown() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    try:
        first_loop = asyncio.run(runtime.run(_current_loop()))
        second_loop = asyncio.run(runtime.run(_current_loop()))
        assert first_loop is second_loop
    finally:
        runtime.stop()

    assert first_loop.is_closed()
    assert not runtime.active


def test_asyncio_loop_thread_closes_resources_on_owned_loop() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    owner_loop = runtime.run_sync(_current_loop())
    cleanup_loops: list[asyncio.AbstractEventLoop] = []

    async def _cleanup() -> None:
        cleanup_loops.append(asyncio.get_running_loop())

    runtime.add_close_callback(_cleanup)
    runtime.stop()

    assert cleanup_loops == [owner_loop]
    assert owner_loop.is_closed()


def test_asyncio_loop_runner_runs_registered_cleanup_before_shutdown() -> None:
    cleanup_loops: list[asyncio.AbstractEventLoop] = []

    async def _run() -> asyncio.AbstractEventLoop:
        owner_loop = asyncio.get_running_loop()

        async def _cleanup() -> None:
            cleanup_loops.append(asyncio.get_running_loop())

        assert AsyncioLoopRunner.register_close_callback(_cleanup)
        return owner_loop

    try:
        owner_loop = AsyncioLoopRunner.run(_run())
    finally:
        AsyncioLoopRunner.shutdown_for_thread()

    assert cleanup_loops == [owner_loop]
    assert owner_loop.is_closed()


def test_asyncio_loop_thread_finalizes_async_generators() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    finalized = threading.Event()
    generators: list[AsyncGenerator[None, None]] = []

    async def _generator() -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            finalized.set()

    async def _start_generator() -> asyncio.AbstractEventLoop:
        generator = _generator()
        generators.append(generator)
        await anext(generator)
        return asyncio.get_running_loop()

    owner_loop = runtime.run_sync(_start_generator())
    runtime.stop()

    assert finalized.is_set()
    assert owner_loop.is_closed()


def test_asyncio_loop_thread_cancels_pending_tasks() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    cancelled = threading.Event()

    async def _wait_forever() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def _start_task() -> asyncio.AbstractEventLoop:
        asyncio.create_task(_wait_forever())
        await asyncio.sleep(0)
        return asyncio.get_running_loop()

    owner_loop = runtime.run_sync(_start_task())
    runtime.stop()

    assert cancelled.is_set()
    assert owner_loop.is_closed()


def test_asyncio_loop_thread_waits_for_default_executor() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    release_worker = threading.Event()
    worker_completed = threading.Event()

    def _work() -> None:
        release_worker.wait()
        worker_completed.set()

    async def _start_work() -> asyncio.AbstractEventLoop:
        asyncio.get_running_loop().run_in_executor(None, _work)
        return asyncio.get_running_loop()

    owner_loop = runtime.run_sync(_start_work())
    threading.Timer(0.05, release_worker.set).start()
    runtime.stop()

    assert worker_completed.is_set()
    assert owner_loop.is_closed()


def test_asyncio_loop_thread_stop_from_owner_runs_callbacks() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    callback_completed = threading.Event()

    async def _stop_from_owner() -> asyncio.AbstractEventLoop:
        owner_loop = asyncio.get_running_loop()

        async def _cleanup() -> None:
            callback_completed.set()

        runtime.add_close_callback(_cleanup)
        runtime.stop()
        return owner_loop

    owner_loop = runtime.run_sync(_stop_from_owner(), timeout=1)
    deadline = time.monotonic() + 1
    while not owner_loop.is_closed() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert callback_completed.is_set()
    assert owner_loop.is_closed()


def test_asyncio_loop_thread_stop_is_idempotent() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    owner_loop = runtime.run_sync(_current_loop())

    runtime.stop()
    runtime.stop()

    assert owner_loop.is_closed()
    assert not runtime.active


def test_asyncio_loop_thread_rejects_restart_until_previous_generation_stops() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    owner_loop = runtime.run_sync(_current_loop())
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def _block_owner_loop() -> None:
        blocker_started.set()
        release_blocker.wait()

    owner_loop.call_soon_threadsafe(_block_owner_loop)
    assert blocker_started.wait(timeout=1)

    stop_thread = threading.Thread(target=runtime.stop)
    stop_thread.start()
    deadline = time.monotonic() + 1
    while runtime.active and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        with pytest.raises(RuntimeError, match='stopping'):
            runtime.run_sync(_current_loop())
    finally:
        release_blocker.set()
        stop_thread.join(timeout=1)

    assert not stop_thread.is_alive()
    assert owner_loop.is_closed()

    cleanup_loops: list[asyncio.AbstractEventLoop] = []

    async def _register_cleanup() -> asyncio.AbstractEventLoop:
        new_loop = asyncio.get_running_loop()

        async def _cleanup() -> None:
            cleanup_loops.append(asyncio.get_running_loop())

        runtime.add_close_callback(_cleanup)
        return new_loop

    new_loop = runtime.run_sync(_register_cleanup())
    runtime.stop()

    assert new_loop is not owner_loop
    assert cleanup_loops == [new_loop]


def test_asyncio_loop_thread_allows_each_callback_its_timeout_budget() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    runtime.run_sync(_current_loop())
    completed: list[str] = []

    async def _cleanup(name: str) -> None:
        await asyncio.sleep(0.15)
        completed.append(name)

    runtime.add_close_callback(lambda: _cleanup('first'))
    runtime.add_close_callback(lambda: _cleanup('second'))
    runtime.stop(join_timeout=0.25)

    assert completed == ['first', 'second']
