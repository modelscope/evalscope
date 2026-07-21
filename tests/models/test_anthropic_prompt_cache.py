import asyncio
import pytest
from pydantic import ValidationError
from types import SimpleNamespace

from evalscope.api.messages import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentText,
)
from evalscope.api.model import GenerateConfig
from evalscope.api.tool import ToolCall, ToolFunction, ToolInfo


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
        GenerateConfig(anthropic_cache_control={'type': 'ephemeral', 'unknown': True})


def test_legacy_anthropic_content_cache_control_is_rejected():
    with pytest.raises(ValidationError, match='anthropic_content_cache_control'):
        GenerateConfig(anthropic_content_cache_control={'type': 'ephemeral'})


def test_completion_params_do_not_enable_cache_by_default():
    anthropic = _anthropic_utils()

    params = anthropic.anthropic_completion_params('claude', GenerateConfig())

    assert 'cache_control' not in params


def test_recent_messages_sets_top_level_cache_control():
    anthropic = _anthropic_utils()

    params = anthropic.anthropic_completion_params(
        'claude',
        GenerateConfig(
            anthropic_cache_control={
                'type': 'ephemeral',
                'ttl': '5m'
            },
            anthropic_cache_strategy='recent_messages',
        ),
    )

    assert params['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}


def test_anthropic_api_splits_automatic_and_explicit_cache_control():
    pytest.importorskip('anthropic')
    from evalscope.models.anthropic_compatible import AnthropicCompatibleAPI

    api = AnthropicCompatibleAPI.__new__(AnthropicCompatibleAPI)
    recent_config = GenerateConfig(
        anthropic_cache_control={'type': 'ephemeral'},
        anthropic_cache_strategy='recent_messages',
    )
    evaluation_config = GenerateConfig(
        anthropic_cache_control={'type': 'ephemeral'},
        anthropic_cache_strategy='evaluation',
    )

    assert api.cache_control_params(recent_config) == {'type': 'ephemeral'}
    assert api.explicit_cache_control_params(recent_config) is None
    assert api.cache_control_params(evaluation_config) == {'type': 'ephemeral'}
    assert api.explicit_cache_control_params(evaluation_config) == {'type': 'ephemeral'}


def test_evaluation_strategy_marks_system_tools_and_fewshot_but_not_final_question():
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

    assert isinstance(system, list)
    assert system[-1]['cache_control'] == cache_control
    assert tools[-1]['cache_control'] == cache_control
    assert 'cache_control' not in tools[0]
    assert messages[-2]['content'][-1]['cache_control'] == cache_control
    assert 'cache_control' not in messages[-1]['content'][-1]
    assert len(_cache_control_blocks(messages)) == 1


def test_recent_messages_uses_top_level_cache_control_without_manual_block_markers():
    anthropic = _anthropic_utils()
    cache_control = {'type': 'ephemeral'}

    system, messages = anthropic.anthropic_chat_messages(
        [
            ChatMessageSystem(content='Stable agent instructions.'),
            ChatMessageUser(content='First turn.'),
            ChatMessageAssistant(content='First answer.'),
            ChatMessageUser(content='Second turn.'),
        ],
        cache_control=cache_control,
        cache_strategy='recent_messages',
    )
    tools = anthropic.anthropic_chat_tools(
        [ToolInfo(name='search', description='Search docs.')],
        cache_control=cache_control,
        cache_strategy='recent_messages',
    )

    assert system == 'Stable agent instructions.'
    assert 'cache_control' not in tools[-1]
    assert len(_cache_control_blocks(messages)) == 0


def test_user_supplied_cache_control_is_preserved_and_not_overwritten():
    anthropic = _anthropic_utils()
    explicit = {'type': 'ephemeral', 'ttl': '1h'}
    automatic = {'type': 'ephemeral', 'ttl': '5m'}

    system, messages = anthropic.anthropic_chat_messages(
        [
            ChatMessageSystem(
                content=[ContentText(text='Stable.', internal={'anthropic': {
                    'cache_control': explicit
                }})]
            ),
            ChatMessageUser(
                content=[ContentText(text='Question.', internal={'anthropic': {
                    'cache_control': explicit
                }})]
            ),
        ],
        cache_control=automatic,
        cache_strategy='recent_messages',
    )

    assert system[-1]['cache_control'] == explicit
    assert messages[-1]['content'][-1]['cache_control'] == explicit


def test_evaluation_cache_control_skips_tool_result_blocks():
    anthropic = _anthropic_utils()

    messages = [
        ChatMessageUser(content='What is the weather?'),
        ChatMessageAssistant(
            content='',
            tool_calls=[ToolCall(id='call_123', function=ToolFunction(name='get_weather', arguments={'city': 'SF'}))],
        ),
        ChatMessageTool(tool_call_id='call_123', content='Sunny, 72F'),
        ChatMessageUser(content='What should I wear?'),
    ]

    system, message_params = anthropic.anthropic_chat_messages(
        messages, cache_control={'type': 'ephemeral'}, cache_strategy='evaluation'
    )

    assert system is None
    assert len(_cache_control_blocks(message_params)) == 1
    assert message_params[0]['content'][-1]['cache_control'] == {'type': 'ephemeral'}

    tool_result_block = [
        block for message in message_params for block in message.get('content', [])
        if isinstance(block, dict) and block.get('type') == 'tool_result'
    ][0]
    assert 'cache_control' not in tool_result_block


def test_tool_result_explicit_cache_control_is_preserved():
    anthropic = _anthropic_utils()
    cache_control = {'type': 'ephemeral', 'ttl': '1h'}

    message = ChatMessageTool(
        tool_call_id='call_123',
        content='Sunny',
        internal={'anthropic': {
            'cache_control': cache_control
        }},
    )

    message_param = anthropic.anthropic_message_param(message)

    assert message_param['content'][0]['cache_control'] == cache_control


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

    message, _ = anthropic.collect_stream_response([
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
