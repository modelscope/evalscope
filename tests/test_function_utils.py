import asyncio
import pytest
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from evalscope.utils import asyncio_runtime
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner, AsyncioLoopThread, cancel_and_wait


async def _current_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


def test_cancel_and_wait_observes_completed_task_failure() -> None:

    async def _run() -> None:

        async def _fail() -> None:
            raise RuntimeError('task failed')

        task = asyncio.create_task(_fail())
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match='task failed'):
            await cancel_and_wait(task)

    asyncio.run(_run())


def test_cancel_and_wait_propagates_waiter_cancellation() -> None:

    async def _run() -> None:
        cleanup_started = asyncio.Event()

        async def _slow_cancel() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await asyncio.Event().wait()

        target = asyncio.create_task(_slow_cancel())
        waiter = asyncio.create_task(cancel_and_wait(target))
        await cleanup_started.wait()

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert target.cancelled()

    asyncio.run(_run())


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
    assert runtime.stop()

    assert callback_completed.is_set()
    assert owner_loop.is_closed()


def test_asyncio_loop_thread_stop_is_idempotent() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    owner_loop = runtime.run_sync(_current_loop())

    runtime.stop()
    runtime.stop()

    assert owner_loop.is_closed()
    assert not runtime.active


def test_asyncio_loop_thread_rejects_restart_until_previous_generation_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    owner_loop = runtime.run_sync(_current_loop())
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    shutdown_scheduled = threading.Event()
    original_schedule_shutdown = AsyncioLoopThread._schedule_shutdown

    def _schedule_shutdown(self, generation) -> None:
        original_schedule_shutdown(self, generation)
        shutdown_scheduled.set()

    monkeypatch.setattr(AsyncioLoopThread, '_schedule_shutdown', _schedule_shutdown)

    def _block_owner_loop() -> None:
        blocker_started.set()
        release_blocker.wait()

    owner_loop.call_soon_threadsafe(_block_owner_loop)
    assert blocker_started.wait(timeout=1)

    stop_thread = threading.Thread(target=runtime.stop)
    stop_thread.start()
    assert shutdown_scheduled.wait(timeout=1)

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


def test_asyncio_loop_thread_allows_each_callback_its_timeout_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    runtime.run_sync(_current_loop())
    completed: list[str] = []
    timeout_budgets: list[float] = []
    original_wait_for = asyncio.wait_for

    async def _cleanup(name: str) -> None:
        completed.append(name)

    async def _record_wait_for(awaitable, timeout):
        timeout_budgets.append(timeout)
        return await original_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(asyncio_runtime.asyncio, 'wait_for', _record_wait_for)
    runtime.add_close_callback(lambda: _cleanup('first'))
    runtime.add_close_callback(lambda: _cleanup('second'))
    runtime.stop(join_timeout=0.25)

    assert completed == ['first', 'second']
    assert timeout_budgets == [0.25, 0.25]


def test_asyncio_loop_thread_immediate_stop_waits_for_owner_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered_run_forever = threading.Event()
    release_run_forever = threading.Event()
    original_run_forever = asyncio.BaseEventLoop.run_forever

    def _delayed_run_forever(loop: asyncio.BaseEventLoop) -> None:
        entered_run_forever.set()
        release_run_forever.wait()
        original_run_forever(loop)

    monkeypatch.setattr(asyncio.BaseEventLoop, 'run_forever', _delayed_run_forever)
    runtime = AsyncioLoopThread(name='DelayedStartLoop')
    owner_loop = runtime.loop
    assert entered_run_forever.wait(timeout=1)

    stop_thread = threading.Thread(target=runtime.stop)
    stop_thread.start()
    release_run_forever.set()
    stop_thread.join(timeout=1)

    assert not stop_thread.is_alive()
    assert owner_loop.is_closed()
    assert not runtime.active


def test_asyncio_loop_thread_drains_callbacks_registered_during_shutdown() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    runtime.run_sync(_current_loop())
    callbacks: list[str] = []

    async def _second_cleanup() -> None:
        callbacks.append('second')

    async def _first_cleanup() -> None:
        callbacks.append('first')
        assert runtime.add_close_callback(_second_cleanup)

    assert runtime.add_close_callback(_first_cleanup)
    runtime.stop()

    assert callbacks == ['first', 'second']

    runtime.run_sync(_current_loop())
    runtime.stop()
    assert callbacks == ['first', 'second']


def test_asyncio_loop_thread_drains_task_finalizer_callbacks_before_loop_close() -> None:
    runtime = AsyncioLoopThread(name='TestLoop')
    order: list[str] = []
    callback_loops: list[asyncio.AbstractEventLoop] = []

    async def _late_cleanup() -> None:
        order.append('late_cleanup')
        callback_loops.append(asyncio.get_running_loop())

    async def _application_task() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            order.append('task_finalizer')
            assert runtime.add_close_callback(_late_cleanup)

    async def _start() -> asyncio.AbstractEventLoop:
        asyncio.create_task(_application_task())
        await asyncio.sleep(0)
        return asyncio.get_running_loop()

    async def _early_cleanup() -> None:
        order.append('early_cleanup')
        callback_loops.append(asyncio.get_running_loop())

    owner_loop = runtime.run_sync(_start())
    assert runtime.add_close_callback(_early_cleanup)
    runtime.stop()

    assert order == ['early_cleanup', 'task_finalizer', 'late_cleanup']
    assert callback_loops == [owner_loop, owner_loop]
    assert owner_loop.is_closed()


def test_asyncio_loop_runner_keeps_timed_out_generation_registered() -> None:
    owner_loop = AsyncioLoopRunner.run(_current_loop())
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def _block_owner_loop() -> None:
        blocker_started.set()
        release_blocker.wait()

    owner_loop.call_soon_threadsafe(_block_owner_loop)
    assert blocker_started.wait(timeout=1)

    AsyncioLoopRunner.shutdown_for_thread(join_timeout=0.01)
    handle = AsyncioLoopRunner._local.handle
    assert handle is not None
    assert handle in AsyncioLoopRunner._all_handles
    assert handle.owns(owner_loop)

    release_blocker.set()
    AsyncioLoopRunner.shutdown_for_thread(join_timeout=1)

    assert AsyncioLoopRunner._local.handle is None
    assert handle not in AsyncioLoopRunner._all_handles
    assert owner_loop.is_closed()


def test_asyncio_loop_runner_keeps_worker_handle_registered_after_shutdown_all() -> None:
    cleanup_loops: list[asyncio.AbstractEventLoop] = []

    def _run(register_cleanup: bool) -> tuple[AsyncioLoopThread, asyncio.AbstractEventLoop, bool]:

        async def _operation() -> tuple[asyncio.AbstractEventLoop, bool]:
            owner_loop = asyncio.get_running_loop()

            async def _cleanup() -> None:
                cleanup_loops.append(asyncio.get_running_loop())

            registered = not register_cleanup or AsyncioLoopRunner.register_close_callback(_cleanup)
            return owner_loop, registered

        owner_loop, registered = AsyncioLoopRunner.run(_operation())
        return AsyncioLoopRunner._local.handle, owner_loop, registered

    with ThreadPoolExecutor(max_workers=1) as executor:
        handle, first_loop, _ = executor.submit(_run, False).result()
        AsyncioLoopRunner.shutdown_all()
        assert first_loop.is_closed()

        reused_handle, second_loop, registered = executor.submit(_run, True).result()
        assert reused_handle is handle
        assert registered
        assert handle in AsyncioLoopRunner._all_handles

        AsyncioLoopRunner.shutdown_all()
        assert second_loop.is_closed()
        assert cleanup_loops == [second_loop]

        executor.submit(AsyncioLoopRunner.shutdown_for_thread).result()

    assert handle not in AsyncioLoopRunner._all_handles
