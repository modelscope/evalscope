import asyncio

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
