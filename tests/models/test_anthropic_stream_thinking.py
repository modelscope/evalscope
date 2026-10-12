import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator, List

import pytest
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    Message,
    MessageStartEvent,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)

from evalscope.api.messages import ChatMessageAssistant, ContentReasoning
from evalscope.models.utils.anthropic import (
    anthropic_message_param,
    async_collect_stream_response,
    chat_message_assistant_from_anthropic,
    collect_stream_response,
)


def _start() -> MessageStartEvent:
    return MessageStartEvent(
        type='message_start',
        message=Message(
            id='msg_thinking', type='message', role='assistant', model='test-model', content=[],
            stop_reason=None, stop_sequence=None, usage=Usage(input_tokens=3, output_tokens=0),
        ),
    )


def _collect(events: List[Any], asynchronous: bool) -> Any:
    if not asynchronous:
        return collect_stream_response(iter(events), request_start=10.0)

    async def stream() -> AsyncIterator[Any]:
        for event in events:
            yield event

    return asyncio.run(async_collect_stream_response(stream(), request_start=10.0))


@pytest.mark.parametrize('asynchronous', [False, True])
def test_stream_preserves_thinking_blocks_and_first_token(
    asynchronous: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        _start(),
        ContentBlockStartEvent(
            type='content_block_start', index=0,
            content_block=ThinkingBlock(type='thinking', thinking='', signature=''),
        ),
        ContentBlockDeltaEvent(type='content_block_delta', index=0, delta={'type': 'thinking_delta', 'thinking': 'Let me '}),
        ContentBlockDeltaEvent(type='content_block_delta', index=0, delta={'type': 'thinking_delta', 'thinking': 'think.'}),
        ContentBlockDeltaEvent(type='content_block_delta', index=0, delta={'type': 'signature_delta', 'signature': 'sig-one'}),
        ContentBlockStartEvent(
            type='content_block_start', index=1,
            content_block=RedactedThinkingBlock(type='redacted_thinking', data='opaque-data'),
        ),
        ContentBlockStartEvent(
            type='content_block_start', index=2,
            content_block=TextBlock(type='text', text=''),
        ),
        ContentBlockDeltaEvent(type='content_block_delta', index=2, delta={'type': 'text_delta', 'text': 'Answer'}),
        ContentBlockStartEvent(
            type='content_block_start', index=3,
            content_block=ToolUseBlock(type='tool_use', id='call_1', name='lookup', input={}),
        ),
        ContentBlockDeltaEvent(
            type='content_block_delta', index=3,
            delta={'type': 'input_json_delta', 'partial_json': '{"query":"value"}'},
        ),
    ]
    times = iter([10.25, 11.0])
    monkeypatch.setattr('evalscope.models.utils.anthropic.time', SimpleNamespace(monotonic=lambda: next(times)))
    message, ttft = _collect(events, asynchronous)
    assert message.content == [
        ThinkingBlock(type='thinking', thinking='Let me think.', signature='sig-one'),
        RedactedThinkingBlock(type='redacted_thinking', data='opaque-data'),
        TextBlock(type='text', text='Answer'),
        ToolUseBlock(type='tool_use', id='call_1', name='lookup', input={'query': 'value'}),
    ]
    assert ttft == 0.25
    assert events[1].content_block == ThinkingBlock(type='thinking', thinking='', signature='')
    assistant = chat_message_assistant_from_anthropic('test-model', message, [])
    assert assistant.content[0] == ContentReasoning(reasoning='Let me think.', signature='sig-one')
    assert assistant.content[1] == ContentReasoning(reasoning='opaque-data', redacted=True)
    assert assistant.text == 'Answer'


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('initial_signature, signatures', [
    ('', ['opaque']),
    ('initial-signature', ['opaque']),
    ('', ['previous-signature', 'opaque']),
])
def test_stream_preserves_empty_signed_thinking(
    asynchronous: bool, initial_signature: str, signatures: List[str],
) -> None:
    message, ttft = _collect([
        _start(),
        ContentBlockStartEvent(
            type='content_block_start', index=0,
            content_block=ThinkingBlock(type='thinking', thinking='', signature=initial_signature),
        ),
        *[ContentBlockDeltaEvent(
            type='content_block_delta', index=0, delta={'type': 'signature_delta', 'signature': signature},
        ) for signature in signatures],
    ], asynchronous)
    assert message.content == [ThinkingBlock(type='thinking', thinking='', signature='opaque')]
    assert ttft is None


@pytest.mark.parametrize('content, expected', [
    (ContentReasoning(reasoning='Thinking', signature='signed'),
     {'type': 'thinking', 'thinking': 'Thinking', 'signature': 'signed'}),
    (ContentReasoning(reasoning='opaque', redacted=True),
     {'type': 'redacted_thinking', 'data': 'opaque'}),
    (ContentReasoning(reasoning='Other provider'),
     {'type': 'text', 'text': '<think>\nOther provider\n</think>'}),
])
def test_assistant_replays_native_thinking(content: ContentReasoning, expected: dict) -> None:
    message = anthropic_message_param(ChatMessageAssistant(content=[content]))
    assert message['content'] == [expected]


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('block', [
    ThinkingBlock(type='thinking', thinking='Initial thought', signature='signature'),
    RedactedThinkingBlock(type='redacted_thinking', data='opaque'),
])
def test_stream_records_thinking_in_start_event(
    asynchronous: bool, block: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr('evalscope.models.utils.anthropic.time', SimpleNamespace(monotonic=lambda: 10.5))
    message, ttft = _collect([
        _start(), ContentBlockStartEvent(type='content_block_start', index=0, content_block=block),
    ], asynchronous)
    assert message.content == [block]
    assert ttft == 0.5
