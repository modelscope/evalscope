import asyncio
from typing import Any, Callable, List

import pytest

from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, Model, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils.function_utils import AsyncioLoopRunner, AsyncioLoopThread


class _FakeAsyncClient:

    def __init__(self, created_clients: List['_FakeAsyncClient'], **kwargs: Any) -> None:
        self.owner_loop = asyncio.get_running_loop()
        self.close_loops: List[asyncio.AbstractEventLoop] = []
        created_clients.append(self)

    async def close(self) -> None:
        self.close_loops.append(asyncio.get_running_loop())


class _ClosableModelAPI(ModelAPI):

    def __init__(self) -> None:
        super().__init__('test-model')
        self.close_count = 0

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        raise NotImplementedError

    async def aclose(self) -> None:
        self.close_count += 1


def _openai_api(monkeypatch: pytest.MonkeyPatch, created_clients: List[_FakeAsyncClient]) -> Any:
    from evalscope.models import openai_compatible

    monkeypatch.setattr(openai_compatible, 'OpenAI', lambda **kwargs: object())
    monkeypatch.setattr(
        openai_compatible,
        'AsyncOpenAI',
        lambda **kwargs: _FakeAsyncClient(created_clients, **kwargs),
    )
    return openai_compatible.OpenAICompatibleAPI(
        model_name='test-model',
        base_url='https://example.test/v1',
        api_key='test-key',
    )


def _anthropic_api(monkeypatch: pytest.MonkeyPatch, created_clients: List[_FakeAsyncClient]) -> Any:
    from evalscope.models import anthropic_compatible

    monkeypatch.setattr(anthropic_compatible, 'Anthropic', lambda **kwargs: object())
    monkeypatch.setattr(
        anthropic_compatible,
        'AsyncAnthropic',
        lambda **kwargs: _FakeAsyncClient(created_clients, **kwargs),
    )
    return anthropic_compatible.AnthropicCompatibleAPI(
        model_name='test-model',
        base_url='https://example.test/v1',
        api_key='test-key',
    )


def test_model_aclose_delegates_to_model_api() -> None:
    api = _ClosableModelAPI()
    model = Model(api, GenerateConfig())

    asyncio.run(model.aclose())

    assert api.close_count == 1


@pytest.mark.parametrize('api_factory', [_openai_api, _anthropic_api])
def test_aclose_releases_and_recreates_client_on_caller_managed_loop(
    monkeypatch: pytest.MonkeyPatch,
    api_factory: Callable[[pytest.MonkeyPatch, List[_FakeAsyncClient]], Any],
) -> None:
    created_clients: List[_FakeAsyncClient] = []
    api = api_factory(monkeypatch, created_clients)

    async def _use_model() -> tuple[_FakeAsyncClient, _FakeAsyncClient]:
        first_client = api.async_client
        await api.aclose()
        second_client = api.async_client
        await api.aclose()
        await api.aclose()
        return first_client, second_client

    first_client, second_client = asyncio.run(_use_model())

    assert first_client is not second_client
    assert first_client.close_loops == [first_client.owner_loop]
    assert second_client.close_loops == [second_client.owner_loop]


def test_aclose_dispatches_cleanup_to_every_client_owner_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: List[_FakeAsyncClient] = []
    api = _openai_api(monkeypatch, created_clients)
    first_runtime = AsyncioLoopThread(name='FirstModelLoop')
    second_runtime = AsyncioLoopThread(name='SecondModelLoop')

    async def _get_client() -> _FakeAsyncClient:
        return api.async_client

    try:
        first_client = first_runtime.run_sync(_get_client())
        second_client = second_runtime.run_sync(_get_client())

        asyncio.run(api.aclose())

        assert first_client.close_loops == [first_client.owner_loop]
        assert second_client.close_loops == [second_client.owner_loop]
    finally:
        first_runtime.stop()
        second_runtime.stop()


def test_runner_shutdown_closes_client_on_runner_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: List[_FakeAsyncClient] = []
    api = _openai_api(monkeypatch, created_clients)

    async def _get_client() -> _FakeAsyncClient:
        return api.async_client

    try:
        client = AsyncioLoopRunner.run(_get_client())
    finally:
        AsyncioLoopRunner.shutdown_for_thread()

    assert client.close_loops == [client.owner_loop]


def test_aclose_rejects_cleanup_after_owner_loop_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: List[_FakeAsyncClient] = []
    api = _openai_api(monkeypatch, created_clients)
    runtime = AsyncioLoopThread(name='StoppedModelLoop')

    async def _get_client() -> _FakeAsyncClient:
        return api.async_client

    client = runtime.run_sync(_get_client())
    runtime.stop()

    with pytest.raises(RuntimeError, match='before shutting down the loop'):
        asyncio.run(api.aclose())

    assert client.close_loops == []
