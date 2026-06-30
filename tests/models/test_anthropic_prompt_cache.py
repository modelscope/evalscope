import asyncio
import pytest
from pydantic import ValidationError
from types import SimpleNamespace

from evalscope.api.messages import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from evalscope.api.model import GenerateConfig
from evalscope.api.tool import ToolInfo


def test_anthropic_cache_control_accepts_typed_values():
    config = GenerateConfig(
        anthropic_cache_control={
            'type': 'ephemeral',
            'ttl': '1h',
        },
        anthropic_cache_strategy='recent_messages',
    )

    assert config.anthropic_cache_control is not None
    assert config.anthropic_cache_control.model_dump(exclude_none=True) == {
        'type': 'ephemeral',
        'ttl': '1h',
    }
    assert config.anthropic_cache_strategy == 'recent_messages'


def test_anthropic_cache_control_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        GenerateConfig(
            anthropic_cache_control={
                'type': 'ephemeral',
                'unknown': True,
            }
        )


def test_legacy_anthropic_content_cache_control_is_rejected():
    with pytest.raises(ValidationError, match='anthropic_content_cache_control'):
        GenerateConfig(anthropic_content_cache_control={'type': 'ephemeral'})


def _anthropic_utils():
    pytest.importorskip('anthropic')
    from evalscope.models.utils import anthropic

    return anthropic


def _cache_control_blocks(messages):
    blocks = []
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            continue
        blocks.extend(block for block in content if isinstance(block, dict) and 'cache_control' in block)
    return blocks


def test_evaluation_strategy_marks_system_and_fewshot_but_not_final_question():
    anthropic = _anthropic_utils()
    cache_control = {'type': 'ephemeral'}

    system, messages = anthropic.anthropic_chat_messages(
        [
            ChatMessageSystem(content='Stable evaluator instructions.'),
            ChatMessageUser(content='Example question.'),
            ChatMessageAssistant(content='Example answer.'),
            ChatMessageUser(content='Current sample question.'),
        ],
        cache_control=cache_control,
        cache_strategy='evaluation',
    )

    assert isinstance(system, list)
    assert system[-1]['cache_control'] == cache_control
    assert messages[-2]['content'][-1]['cache_control'] == cache_control
    assert 'cache_control' not in messages[-1]['content'][-1]
    assert len(_cache_control_blocks(messages)) == 1


def test_evaluation_strategy_is_noop_without_stable_message_prefix():
    anthropic = _anthropic_utils()

    system, messages = anthropic.anthropic_chat_messages(
        [ChatMessageUser(content='Current sample question.')],
        cache_control={'type': 'ephemeral'},
        cache_strategy='evaluation',
    )

    assert system is None
    assert messages[0]['content'] == 'Current sample question.'
    assert _cache_control_blocks(messages) == []


def test_recent_messages_strategy_marks_only_one_tail_block():
    anthropic = _anthropic_utils()
    cache_control = {'type': 'ephemeral'}

    system, messages = anthropic.anthropic_chat_messages(
        [
            ChatMessageSystem(content='Stable evaluator instructions.'),
            ChatMessageUser(content='First turn.'),
            ChatMessageAssistant(content='First answer.'),
            ChatMessageUser(content='Second turn.'),
        ],
        cache_control=cache_control,
        cache_strategy='recent_messages',
    )

    assert system == 'Stable evaluator instructions.'
    assert len(_cache_control_blocks(messages)) == 1
    assert messages[-1]['content'][-1]['cache_control'] == cache_control
    assert 'cache_control' not in messages[-2]['content'][-1]


def test_cache_control_none_leaves_message_shape_unchanged():
    anthropic = _anthropic_utils()

    system, messages = anthropic.anthropic_chat_messages(
        [
            ChatMessageSystem(content='Stable evaluator instructions.'),
            ChatMessageUser(content='Current sample question.'),
        ],
        cache_control=None,
    )

    assert system == 'Stable evaluator instructions.'
    assert messages == [{
        'role': 'user',
        'content': 'Current sample question.',
    }]


def test_tool_cache_control_marks_only_last_tool_in_evaluation_strategy():
    anthropic = _anthropic_utils()
    cache_control = {'type': 'ephemeral'}

    tools = anthropic.anthropic_chat_tools(
        [
            ToolInfo(name='search', description='Search docs.'),
            ToolInfo(name='read', description='Read docs.'),
        ],
        cache_control=cache_control,
        cache_strategy='evaluation',
    )

    assert 'cache_control' not in tools[0]
    assert tools[1]['cache_control'] == cache_control


def test_evaluation_strategy_stays_under_anthropic_breakpoint_limit():
    anthropic = _anthropic_utils()
    cache_control = {'type': 'ephemeral'}

    system, messages = anthropic.anthropic_chat_messages(
        [
            ChatMessageSystem(content='Stable evaluator instructions.'),
            ChatMessageUser(content='Example question.'),
            ChatMessageAssistant(content='Example answer.'),
            ChatMessageUser(content='Current sample question.'),
        ],
        cache_control=cache_control,
        cache_strategy='evaluation',
    )
    tools = anthropic.anthropic_chat_tools(
        [
            ToolInfo(name='search', description='Search docs.'),
            ToolInfo(name='read', description='Read docs.'),
        ],
        cache_control=cache_control,
        cache_strategy='evaluation',
    )

    system_blocks = [block for block in system if 'cache_control' in block]
    tool_blocks = [tool for tool in tools if 'cache_control' in tool]
    message_blocks = _cache_control_blocks(messages)
    breakpoint_count = len(system_blocks) + len(tool_blocks) + len(message_blocks)

    assert breakpoint_count == 3
    assert breakpoint_count <= anthropic.MAX_ANTHROPIC_CACHE_CONTROL_BLOCKS


class _FakeUsage:

    def __init__(
        self,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens

    def model_dump(self):
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cache_creation_input_tokens': self.cache_creation_input_tokens,
            'cache_read_input_tokens': self.cache_read_input_tokens,
        }


class _FakeMessage:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeMessageStartEvent:

    def __init__(self, message):
        self.message = message


class _FakeMessageDeltaEvent:

    def __init__(self, usage):
        self.delta = SimpleNamespace(stop_reason='end_turn')
        self.usage = usage


class _FakeContentBlockStartEvent:
    pass


class _FakeContentBlockDeltaEvent:
    pass


def _patch_stream_event_types(monkeypatch, anthropic_utils):
    import anthropic.types as anthropic_types

    monkeypatch.setattr(anthropic_utils, 'Message', _FakeMessage)
    monkeypatch.setattr(anthropic_types, 'MessageStartEvent', _FakeMessageStartEvent)
    monkeypatch.setattr(anthropic_types, 'MessageDeltaEvent', _FakeMessageDeltaEvent)
    monkeypatch.setattr(anthropic_types, 'ContentBlockStartEvent', _FakeContentBlockStartEvent)
    monkeypatch.setattr(anthropic_types, 'ContentBlockDeltaEvent', _FakeContentBlockDeltaEvent)
    monkeypatch.setattr(anthropic_types, 'Usage', _FakeUsage)


def test_collect_stream_response_preserves_cache_usage(monkeypatch):
    anthropic = _anthropic_utils()
    _patch_stream_event_types(monkeypatch, anthropic)

    message, _ = anthropic.collect_stream_response(
        [
            _FakeMessageStartEvent(
                _FakeMessage(
                    id='msg_1',
                    model='claude',
                    role='assistant',
                    usage=_FakeUsage(
                        input_tokens=10,
                        cache_creation_input_tokens=3,
                        cache_read_input_tokens=4,
                    ),
                )
            ),
            _FakeMessageDeltaEvent(_FakeUsage(output_tokens=2)),
        ]
    )

    assert message.usage.input_tokens == 10
    assert message.usage.output_tokens == 2
    assert message.usage.cache_creation_input_tokens == 3
    assert message.usage.cache_read_input_tokens == 4


async def _fake_async_events(events):
    for event in events:
        yield event


def test_async_collect_stream_response_preserves_cache_usage(monkeypatch):
    anthropic = _anthropic_utils()
    _patch_stream_event_types(monkeypatch, anthropic)

    message, _ = asyncio.run(
        anthropic.async_collect_stream_response(
            _fake_async_events([
                _FakeMessageStartEvent(
                    _FakeMessage(
                        id='msg_1',
                        model='claude',
                        role='assistant',
                        usage=_FakeUsage(
                            input_tokens=10,
                            cache_creation_input_tokens=3,
                            cache_read_input_tokens=4,
                        ),
                    )
                ),
                _FakeMessageDeltaEvent(_FakeUsage(output_tokens=2)),
            ])
        )
    )

    assert message.usage.input_tokens == 10
    assert message.usage.output_tokens == 2
    assert message.usage.cache_creation_input_tokens == 3
    assert message.usage.cache_read_input_tokens == 4
