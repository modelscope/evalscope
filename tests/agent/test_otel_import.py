"""OTLP GenAI trace importer.

Covers the mapping matrix from the feature issue: operation → EventType,
``gen_ai.usage.*`` → token_usage, ``gen_ai.*.messages`` → ChatMessage,
unknown attributes preserved in payload, and the fail-loud paths (missing
operation name, missing message content, unresolvable sample id).
"""
import pytest

from evalscope.api.agent.otel import FRAMEWORK, OTLPImportError, import_otlp_trace
from evalscope.api.agent.trace import EventType
from evalscope.api.messages import ChatMessageAssistant, ChatMessageTool, ChatMessageUser


def _str(value: str) -> dict:
    return {'stringValue': value}


def _int(value: int) -> dict:
    return {'intValue': value}


def _span(
    span_id: str,
    name: str,
    *,
    parent: str = '',
    start: int = 0,
    end: int = 0,
    attributes: list | None = None,
) -> dict:
    return {
        'spanId': span_id,
        'parentSpanId': parent,
        'name': name,
        'startTimeUnixNano': str(start),
        'endTimeUnixNano': str(end),
        'attributes': attributes or [],
    }


def _export(*spans: dict) -> dict:
    return {'resourceSpans': [{'scopeSpans': [{'spans': list(spans)}]}]}


def _chat_attributes(prompt: int = 10, completion: int = 5, **extra) -> list:
    attrs = [
        {'key': 'gen_ai.operation.name', 'value': _str('chat')},
        {'key': 'gen_ai.request.model', 'value': _str('some-model')},
        {'key': 'gen_ai.usage.input_tokens', 'value': _int(prompt)},
        {'key': 'gen_ai.usage.output_tokens', 'value': _int(completion)},
    ]
    attrs.extend({'key': key, 'value': value} for key, value in extra.items())
    return attrs


def _any_value(value):
    """Encode a Python value as an OTel AnyValue (dicts → kvlist, lists → array)."""
    if isinstance(value, bool):
        return {'boolValue': value}
    if isinstance(value, int):
        return _int(value)
    if isinstance(value, float):
        return {'doubleValue': value}
    if isinstance(value, dict):
        return {'kvlistValue': {'values': _kv_list(value)}}
    if isinstance(value, list):
        return {'arrayValue': {'values': [_any_value(item) for item in value]}}
    return _str(str(value))


def _kv_list(obj: dict) -> list:
    return [{'key': key, 'value': _any_value(value)} for key, value in obj.items()]


def _messages_attr(*messages: dict) -> dict:
    return _any_value(list(messages))


def _text_part(content: str) -> dict:
    return {'type': 'text', 'content': content}


def test_single_chat_span_maps_to_model_generate_with_usage():
    payload = _export(
        _span(
            's1',
            'chat',
            start=1,
            end=1_500_000,
            attributes=_chat_attributes(
                prompt=12,
                completion=8,
                **{
                    'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('hi')]}),
                    'gen_ai.output.messages': _messages_attr(
                        {'role': 'assistant', 'parts': [_text_part('hello')]}
                    ),
                },
            ),
        )
    )

    trace, messages = import_otlp_trace(payload)

    assert trace.framework == FRAMEWORK
    assert [e.type for e in trace.events] == [EventType.MODEL_GENERATE]
    assert trace.events[0].token_usage == {'input': 12, 'output': 8, 'total': 20}
    assert trace.events[0].latency_ms == pytest.approx(1.5)
    assert isinstance(messages[0], ChatMessageUser)
    assert messages[0].text == 'hi'
    assert isinstance(messages[1], ChatMessageAssistant)
    assert messages[1].text == 'hello'


def test_tool_span_maps_to_tool_call_and_result():
    payload = _export(
        _span(
            'root',
            'invoke_agent',
            start=0,
            end=3_000_000,
            attributes=[{'key': 'gen_ai.operation.name', 'value': _str('invoke_agent')}],
        ),
        _span(
            'chat1',
            'chat',
            parent='root',
            start=1,
            end=1_000_000,
            attributes=_chat_attributes(
                **{
                    'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('do it')]}),
                    'gen_ai.output.messages': _messages_attr(
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'tool_call', 'id': 'call_1', 'name': 'search', 'arguments': {'q': 'x'}}],
                        }
                    ),
                }
            ),
        ),
        _span(
            'tool1',
            'search',
            parent='chat1',
            start=1_100_000,
            end=1_200_000,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('execute_tool')},
                {'key': 'gen_ai.tool.name', 'value': _str('search')},
                {'key': 'gen_ai.tool.call.id', 'value': _str('call_1')},
                {'key': 'gen_ai.tool.call.result', 'value': _str('found')},
            ],
        ),
    )

    trace, messages = import_otlp_trace(payload)
    types = [e.type for e in trace.events]

    assert EventType.RUN_START in types and EventType.RUN_END in types
    tool_call = next(e for e in trace.events if e.type is EventType.TOOL_CALL)
    assert tool_call.payload['name'] == 'search'
    assert tool_call.payload['tool_call_id'] == 'call_1'
    assert tool_call.step == 2  # nested under chat1 under root
    tool_result = next(e for e in trace.events if e.type is EventType.TOOL_RESULT)
    assert tool_result.payload['tool_call_id'] == 'call_1'

    assistant = next(m for m in messages if isinstance(m, ChatMessageAssistant) and m.tool_calls)
    assert assistant.tool_calls[0].function.name == 'search'
    assert assistant.tool_calls[0].function.arguments == {'q': 'x'}


def test_unknown_attribute_survives_in_payload():
    payload = _export(
        _span(
            's1',
            'chat',
            start=1,
            end=2,
            attributes=_chat_attributes(
                **{
                    'vendor.custom.flag': _str('kept'),
                    'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('hi')]}),
                    'gen_ai.output.messages': _messages_attr(
                        {'role': 'assistant', 'parts': [_text_part('yo')]}
                    ),
                }
            ),
        )
    )

    trace, _ = import_otlp_trace(payload)
    assert trace.events[0].payload['vendor.custom.flag'] == 'kept'


def test_unknown_operation_is_kept_visible_not_silently_dropped():
    payload = _export(
        _span(
            's1',
            'chat',
            start=1,
            end=2,
            attributes=_chat_attributes(
                **{
                    'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('hi')]}),
                    'gen_ai.output.messages': _messages_attr(
                        {'role': 'assistant', 'parts': [_text_part('yo')]}
                    ),
                }
            ),
        ),
        _span(
            's2',
            'weird',
            start=3,
            end=4,
            attributes=[{'key': 'gen_ai.operation.name', 'value': _str('embedding_v2')}],
        ),
    )

    trace, _ = import_otlp_trace(payload)
    unsupported = [e for e in trace.events if e.payload.get('unsupported_operation')]
    assert len(unsupported) == 1
    assert unsupported[0].payload['operation'] == 'embedding_v2'


def test_missing_message_content_is_rejected_with_span_id():
    # Metadata-only export (content capture disabled) must not import as an empty trace.
    payload = _export(_span('span-42', 'chat', start=1, end=2, attributes=_chat_attributes()))

    with pytest.raises(OTLPImportError, match='content capture'):
        import_otlp_trace(payload)


def test_missing_operation_name_is_rejected_with_span_id():
    payload = _export(_span('span-7', 'anonymous', start=1, end=2, attributes=[]))

    with pytest.raises(OTLPImportError, match='span-7'):
        import_otlp_trace(payload)


def test_empty_export_is_rejected():
    with pytest.raises(OTLPImportError, match='no spans'):
        import_otlp_trace({'resourceSpans': []})


def test_no_chat_span_is_rejected():
    payload = _export(
        _span(
            's1',
            'tool',
            start=1,
            end=2,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('execute_tool')},
                {'key': 'gen_ai.tool.name', 'value': _str('run')},
            ],
        )
    )

    with pytest.raises(OTLPImportError, match='No .chat. spans'):
        import_otlp_trace(payload)


def test_sample_association_by_attribute():
    def chat(span_id: str, sample_id: str) -> dict:
        return _span(
            span_id,
            'chat',
            start=1,
            end=2,
            attributes=_chat_attributes(
                **{
                    'evalscope.sample_id': _str(sample_id),
                    'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part(f'q-{sample_id}')]}),
                    'gen_ai.output.messages': _messages_attr(
                        {'role': 'assistant', 'parts': [_text_part(f'a-{sample_id}')]}
                    ),
                }
            ),
        )

    payload = _export(chat('s1', 'sample-a'), chat('s2', 'sample-b'))

    trace, messages = import_otlp_trace(payload, expected_sample_id='sample-b')

    assert len(trace.events) == 1
    assert messages[0].text == 'q-sample-b'


def test_unresolvable_sample_id_names_the_missing_value():
    payload = _export(
        _span(
            's1',
            'chat',
            start=1,
            end=2,
            attributes=_chat_attributes(
                **{
                    'evalscope.sample_id': _str('sample-a'),
                    'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('hi')]}),
                    'gen_ai.output.messages': _messages_attr(
                        {'role': 'assistant', 'parts': [_text_part('yo')]}
                    ),
                }
            ),
        )
    )

    with pytest.raises(OTLPImportError, match='sample-zzz'):
        import_otlp_trace(payload, expected_sample_id='sample-zzz')


def test_tool_result_message_is_reconstructed():
    payload = _export(
        _span(
            's1',
            'chat',
            start=1,
            end=2,
            attributes=_chat_attributes(
                **{
                    'gen_ai.input.messages': _messages_attr(
                        {'role': 'user', 'parts': [_text_part('search')]},
                        {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': 'c1', 'name': 'search', 'arguments': {}}]},
                        {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'c1', 'content': 'result'}]},
                    ),
                    'gen_ai.output.messages': _messages_attr(
                        {'role': 'assistant', 'parts': [_text_part('done')]}
                    ),
                }
            ),
        )
    )

    _, messages = import_otlp_trace(payload)

    tool_messages = [m for m in messages if isinstance(m, ChatMessageTool)]
    assert len(tool_messages) == 1
    assert tool_messages[0].text == 'result'
    assert tool_messages[0].tool_call_id == 'c1'


def test_legacy_usage_aliases_are_accepted():
    """Pre-2025 exports used prompt_tokens / completion_tokens."""
    payload = _export(
        _span(
            's1',
            'chat',
            start=1,
            end=2,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('chat')},
                {'key': 'gen_ai.usage.prompt_tokens', 'value': _int(3)},
                {'key': 'gen_ai.usage.completion_tokens', 'value': _int(4)},
                {'key': 'gen_ai.input.messages', 'value': _messages_attr({'role': 'user', 'parts': [_text_part('hi')]})},
                {
                    'key': 'gen_ai.output.messages',
                    'value': _messages_attr({'role': 'assistant', 'parts': [_text_part('yo')]}),
                },
            ],
        )
    )

    trace, _ = import_otlp_trace(payload)
    assert trace.events[0].token_usage == {'input': 3, 'output': 4, 'total': 7}
