import asyncio
import pytest
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import AsyncMock

from evalscope.agent.external.bridge import ModelProxyServer
from evalscope.api.model import GenerateConfig
from evalscope.utils.function_utils import AsyncioLoopRunner


@pytest.fixture(autouse=True)
def _clear_bridge_runner() -> Iterator[None]:
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def test_registry_uses_owner_loop_and_explicit_shutdown_clears_instance() -> None:
    async def run() -> None:
        owner_loop = asyncio.get_running_loop()
        proxy = await ModelProxyServer.get_or_start()

        assert ModelProxyServer._instances == {owner_loop: proxy}
        assert proxy._owner_loop is owner_loop

        await proxy.shutdown()
        assert ModelProxyServer._instances == {}

    asyncio.run(run())


def test_runner_shutdown_clears_bridge_instance() -> None:
    async def start() -> asyncio.AbstractEventLoop:
        owner_loop = asyncio.get_running_loop()
        proxy = await ModelProxyServer.get_or_start()
        assert ModelProxyServer._instances == {owner_loop: proxy}
        return owner_loop

    owner_loop = AsyncioLoopRunner.run(start())
    AsyncioLoopRunner.shutdown_for_thread()

    assert owner_loop.is_closed()
    assert ModelProxyServer._instances == {}


def test_shutdown_rejects_non_owner_loop() -> None:
    async def start() -> ModelProxyServer:
        return await ModelProxyServer.get_or_start()

    proxy = AsyncioLoopRunner.run(start())
    with pytest.raises(RuntimeError, match='owning event loop'):
        asyncio.run(proxy.shutdown())

    AsyncioLoopRunner.shutdown_for_thread()
    assert ModelProxyServer._instances == {}


def test_cancelled_start_waiter_does_not_orphan_shared_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    start_entered = asyncio.Event()
    release_start = asyncio.Event()

    async def slow_start(self: ModelProxyServer) -> None:
        start_entered.set()
        await release_start.wait()
        self._started = True

    async def run() -> None:
        owner_loop = asyncio.get_running_loop()
        first = asyncio.create_task(ModelProxyServer.get_or_start())
        second = asyncio.create_task(ModelProxyServer.get_or_start())
        await start_entered.wait()

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        release_start.set()
        proxy = await second
        assert ModelProxyServer._instances == {owner_loop: proxy}
        assert await ModelProxyServer.get_or_start() is proxy

        await proxy.shutdown()

    monkeypatch.setattr(ModelProxyServer, '_start_server', slow_start)
    asyncio.run(run())


def test_start_failure_cleans_runner_and_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = SimpleNamespace(setup=AsyncMock(), cleanup=AsyncMock())

    class FailingSite:

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def start(self) -> None:
            raise OSError('bind failed')

    async def run() -> None:
        with pytest.raises(OSError, match='bind failed'):
            await ModelProxyServer.get_or_start()
        assert ModelProxyServer._instances == {}

    monkeypatch.setattr('evalscope.agent.external.bridge.server.web.AppRunner', lambda *args, **kwargs: runner)
    monkeypatch.setattr('evalscope.agent.external.bridge.server.web.TCPSite', FailingSite)
    asyncio.run(run())

    runner.cleanup.assert_awaited_once()


@pytest.mark.parametrize('protocol', ['openai', 'anthropic', 'gemini'])
def test_stream_disconnect_waits_for_generation_task(
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
) -> None:
    generation_started = asyncio.Event()
    generation_finished = asyncio.Event()

    class BlockingModel:

        async def generate_async(self, **kwargs: Any) -> None:
            generation_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                generation_finished.set()

    class FakeResponse:

        async def write(self, chunk: bytes) -> None:
            await asyncio.sleep(0)

        async def write_eof(self) -> None:
            return None

    async def prepare_response(request: Any) -> FakeResponse:
        return FakeResponse()

    async def run() -> None:
        owner_loop = asyncio.get_running_loop()
        proxy = ModelProxyServer('127.0.0.1', None, owner_loop)
        session = SimpleNamespace(
            model=BlockingModel(),
            framework='test',
            trial_id='test-trial',
            recorder=None,
        )
        body = {'model': 'test-model'}
        config = GenerateConfig()

        if protocol == 'openai':
            response_coro = proxy._respond_streaming_openai(
                None, session, body, [], [], None, config, include_usage=False
            )
        elif protocol == 'gemini':
            response_coro = proxy._respond_streaming_gemini(None, session, body, [], [], None, config)
        else:
            response_coro = proxy._respond_streaming(None, session, body, [], [], config)

        handler_task = asyncio.create_task(response_coro)
        await generation_started.wait()
        handler_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handler_task

        assert handler_task.done()
        assert generation_finished.is_set()
        current_task = asyncio.current_task()
        assert all(task is current_task or task.done() for task in asyncio.all_tasks())

    monkeypatch.setattr(ModelProxyServer, '_prepare_sse_response', staticmethod(prepare_response))
    asyncio.run(run())


def test_stream_write_disconnect_waits_for_generation_task(monkeypatch: pytest.MonkeyPatch) -> None:
    generation_finished = asyncio.Event()

    class BlockingModel:

        async def generate_async(self, **kwargs: Any) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                generation_finished.set()

    class DisconnectedResponse:

        async def write(self, chunk: bytes) -> None:
            await asyncio.sleep(0)
            raise ConnectionResetError

        async def write_eof(self) -> None:
            return None

    async def prepare_response(request: Any) -> DisconnectedResponse:
        return DisconnectedResponse()

    async def run() -> None:
        proxy = ModelProxyServer('127.0.0.1', None, asyncio.get_running_loop())
        session = SimpleNamespace(
            model=BlockingModel(),
            framework='test',
            trial_id='test-trial',
            recorder=None,
        )
        await proxy._respond_streaming_openai(
            None,
            session,
            {'model': 'test-model'},
            [],
            [],
            None,
            GenerateConfig(),
            include_usage=False,
        )
        assert generation_finished.is_set()
        current_task = asyncio.current_task()
        assert all(task is current_task or task.done() for task in asyncio.all_tasks())

    monkeypatch.setattr(ModelProxyServer, '_prepare_sse_response', staticmethod(prepare_response))
    asyncio.run(run())
