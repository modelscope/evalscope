import asyncio
import json

from evalscope.agent.external.bridge.server import _build_generate_config
from evalscope.agent.external.bridge.sse_anthropic import stream_anthropic_response
from evalscope.agent.external.bridge.translate_anthropic import (
    anthropic_request_to_messages,
    anthropic_tools_to_tool_infos,
)
from evalscope.api.model import ModelOutput, ModelUsage


def _cache_control(ttl='5m'):
    return {'type': 'ephemeral', 'ttl': ttl}


def test_anthropic_bridge_top_level_cache_control_enables_recent_messages_strategy():
    config = _build_generate_config({
        'max_tokens': 1024,
        'cache_control': _cache_control('1h'),
        'messages': [{
            'role': 'user',
            'content': 'hi',
        }],
    })

    assert config.anthropic_cache_control is not None
    assert config.anthropic_cache_control.model_dump(exclude_none=True) == _cache_control('1h')
    assert config.anthropic_cache_strategy == 'recent_messages'
    assert config.extra_body is None


def test_anthropic_bridge_top_level_cache_control_does_not_auto_mark_explicit_blocks():
    config = _build_generate_config({
        'cache_control': _cache_control('1h'),
        'messages': [{
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': 'hi',
                'cache_control': _cache_control(),
            }],
        }],
    })

    assert config.anthropic_cache_control is None
    assert config.anthropic_cache_strategy == 'evaluation'
    assert config.extra_body is None


def test_anthropic_bridge_tool_schema_cache_control_property_does_not_disable_top_level_cache():
    config = _build_generate_config({
        'cache_control': _cache_control('1h'),
        'messages': [{
            'role': 'user',
            'content': 'hi',
        }],
        'tools': [{
            'name': 'configure',
            'description': 'Configure cache settings.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'cache_control': {'type': 'string'},
                },
            },
        }],
    })

    assert config.anthropic_cache_control is not None
    assert config.anthropic_cache_control.model_dump(exclude_none=True) == _cache_control('1h')
    assert config.anthropic_cache_strategy == 'recent_messages'


def test_anthropic_bridge_preserves_block_level_cache_control():
    body = {
        'system': [{
            'type': 'text',
            'text': 'stable system',
            'cache_control': _cache_control('1h'),
        }],
        'messages': [
            {
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': 'cached user',
                    'cache_control': _cache_control(),
                }],
            },
            {
                'role': 'assistant',
                'content': [{
                    'type': 'tool_use',
                    'id': 'toolu_1',
                    'name': 'search',
                    'input': {'q': 'x'},
                    'cache_control': _cache_control('1h'),
                }],
            },
            {
                'role': 'user',
                'content': [{
                    'type': 'tool_result',
                    'tool_use_id': 'toolu_1',
                    'content': 'result',
                    'cache_control': _cache_control(),
                }],
            },
        ],
        'tools': [{
            'name': 'search',
            'description': 'Search docs.',
            'input_schema': {'type': 'object', 'properties': {}, 'required': []},
            'cache_control': _cache_control('1h'),
        }],
    }

    messages = anthropic_request_to_messages(body)
    tools = anthropic_tools_to_tool_infos(body['tools'])

    assert messages[0].role == 'system'
    assert messages[0].content[0].internal == {'anthropic': {'cache_control': _cache_control('1h')}}
    assert messages[1].role == 'user'
    assert messages[1].content[0].internal == {'anthropic': {'cache_control': _cache_control()}}
    assert messages[2].role == 'assistant'
    assert messages[2].tool_calls[0].internal == {'anthropic': {'cache_control': _cache_control('1h')}}
    assert messages[3].role == 'tool'
    assert messages[3].internal == {'anthropic': {'cache_control': _cache_control()}}
    assert tools[0].options == {'anthropic': {'cache_control': _cache_control('1h')}}


def test_anthropic_bridge_sse_usage_includes_cache_tokens():

    async def _go():
        output = ModelOutput.from_content(model='claude', content='ok')
        output.usage = ModelUsage(
            input_tokens=10,
            output_tokens=2,
            total_tokens=19,
            input_tokens_cache_write=3,
            input_tokens_cache_read=4,
        )
        task = asyncio.create_task(asyncio.sleep(0, result=output))
        events = []
        async for chunk in stream_anthropic_response(task, request_model='claude'):
            text = chunk.decode('utf-8')
            if text.startswith('event: message_delta'):
                data = text.split('data: ', 1)[1].strip()
                events.append(json.loads(data))
        return events

    message_delta = asyncio.run(_go())[0]

    assert message_delta['usage'] == {
        'input_tokens': 10,
        'output_tokens': 2,
        'cache_creation_input_tokens': 3,
        'cache_read_input_tokens': 4,
    }
