import asyncio
from types import SimpleNamespace
from typing import Iterator, List, Optional

import pytest
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    Message,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageStartEvent,
    TextBlock,
    TextDelta,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta

from evalscope.api.model import GenerateConfig
from evalscope.models.anthropic_compatible import AnthropicCompatibleAPI


def _events(text: str) -> List:
    """Build a minimal but valid Anthropic streaming event sequence yielding `text`."""
    return [
        MessageStartEvent(
            type='message_start',
            message=Message(
                id='message-id',
                type='message',
                role='assistant',
                model='test-model',
                content=[],
                stop_reason=None,
                stop_sequence=None,
                usage=Usage(input_tokens=3, output_tokens=0),
            ),
        ),
        ContentBlockStartEvent(type='content_block_start', index=0, content_block=TextBlock(type='text', text='')),
        ContentBlockDeltaEvent(type='content_block_delta', index=0, delta=TextDelta(type='text_delta', text=text)),
        MessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn', stop_sequence=None),
            usage=MessageDeltaUsage(output_tokens=5),
        ),
    ]


def _prepare_api(monkeypatch: pytest.MonkeyPatch) -> AnthropicCompatibleAPI:
    api = object.__new__(AnthropicCompatibleAPI)
    api.model_name = 'test-model'
    api.resolve_tools = lambda tools, tool_choice, config: (tools, tool_choice, config)
    api.completion_params = lambda config: {'model': 'test-model', 'stream': True}
    api.explicit_cache_control_params = lambda config: None
    api.validate_request_params = lambda request: None
    api.on_response = lambda response: None
    api.chat_choices_from_message = lambda message, tools: []

    monkeypatch.setattr('evalscope.models.anthropic_compatible.anthropic_chat_messages', lambda *a, **kw: (None, []))

    def model_output(message, choices):
        return SimpleNamespace(
            content=message.content[0].text,
            usage=None,
            message=SimpleNamespace(),
            time=None,
        )

    monkeypatch.setattr('evalscope.models.anthropic_compatible.model_output_from_anthropic', model_output)
    return api


def test_generate_retries_when_stream_consumption_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _prepare_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        attempt = attempts

        def stream() -> Iterator:
            if attempt == 1:
                yield _events('discarded partial response')[2]
                raise ConnectionError('stream interrupted by upstream gateway')
            yield from _events('complete response')

        return stream()

    api.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = api.generate([], [], None, GenerateConfig(retries=2, retry_interval=0, stream=True))

    assert attempts == 2
    assert result.content == 'complete response'


def test_generate_async_retries_when_stream_consumption_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _prepare_api(monkeypatch)
    attempts = 0

    async def create(**request):
        nonlocal attempts
        attempts += 1
        attempt = attempts

        async def stream():
            if attempt == 1:
                yield _events('discarded partial response')[2]
                raise ConnectionError('stream interrupted by upstream gateway')
            for event in _events('complete response'):
                yield event

        return stream()

    async_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(AnthropicCompatibleAPI, 'async_client', property(lambda self: async_client))

    result: Optional[SimpleNamespace] = asyncio.run(
        api.generate_async([], [], None, GenerateConfig(retries=2, retry_interval=0, stream=True))
    )

    assert attempts == 2
    assert result.content == 'complete response'
