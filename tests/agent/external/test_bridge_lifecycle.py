import asyncio
import pytest
from types import SimpleNamespace
from typing import Any, Iterator

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
