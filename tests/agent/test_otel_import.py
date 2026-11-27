"""OTLP GenAI trace importer.

Covers the mapping matrix from the feature issue: operation → EventType,
``gen_ai.usage.*`` → token_usage, ``gen_ai.*.messages`` → ChatMessage,
``step`` as the turn index (not span depth), ``create_agent`` without a run
boundary, skipped-but-counted operations, and the fail-loud paths (missing
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
    status: dict | None = None,
    events: list | None = None,
) -> dict:
    span = {
        'spanId': span_id,
        'parentSpanId': parent,
        'name': name,
        'startTimeUnixNano': str(start),
        'endTimeUnixNano': str(end),
        'attributes': attributes or [],
    }
    if status is not None:
        span['status'] = status
    if events is not None:
        span['events'] = events
    return span


def _export(*spans: dict) -> dict:
    return {'resourceSpans': [{'scopeSpans': [{'spans': list(spans)}]}]}


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


def _chat_attributes(prompt: int = 10, completion: int = 5, **extra) -> list:
    attrs = [
        {'key': 'gen_ai.operation.name', 'value': _str('chat')},
        {'key': 'gen_ai.request.model', 'value': _str('some-model')},
        {'key': 'gen_ai.usage.input_tokens', 'value': _int(prompt)},
        {'key': 'gen_ai.usage.output_tokens', 'value': _int(completion)},
    ]
    attrs.extend({'key': key, 'value': value} for key, value in extra.items())
    return attrs


def _chat_span(span_id: str, *, parent: str = '', start: int = 0, end: int = 1_000_000, **extra) -> dict:
    return _span(span_id, 'chat', parent=parent, start=start, end=end, attributes=_chat_attributes(**extra))


def test_single_chat_span_maps_to_model_generate_with_usage():
    payload = _export(
        _chat_span(
            's1',
            start=1,
            end=1_500_000,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('hi')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('hello')]}),
            },
        )
    )

    trace, messages = import_otlp_trace(payload)

    assert trace.framework == FRAMEWORK
    assert [e.type for e in trace.events] == [EventType.MODEL_GENERATE]
    assert trace.events[0].token_usage == {'input': 10, 'output': 5, 'total': 15}
    assert trace.events[0].latency_ms == pytest.approx(1.5)
    assert isinstance(messages[0], ChatMessageUser) and messages[0].text == 'hi'
    assert isinstance(messages[1], ChatMessageAssistant) and messages[1].text == 'hello'


def test_step_is_turn_index_not_span_depth():
    """Two chat turns both nested under invoke_agent must yield steps 0 and 1."""
    payload = _export(
        _span('root', 'invoke_agent', start=0, end=10_000_000, attributes=[{'key': 'gen_ai.operation.name', 'value': _str('invoke_agent')}]),
        _chat_span(
            'c1',
            parent='root',
            start=1,
            end=2_000_000,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q1')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a1')]}),
            },
        ),
        _chat_span(
            'c2',
            parent='root',
            start=3,
            end=4_000_000,
            **{
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a2')]}),
            },
        ),
    )

    trace, _ = import_otlp_trace(payload)
    chat_steps = [e.step for e in trace.events if e.type is EventType.MODEL_GENERATE]

    assert chat_steps == [0, 1]
    assert trace.step_count == 2, 'step_count must count turns, not nesting depth'


def test_tool_span_inherits_its_turn_and_dedups_against_messages():
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2_000_000,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('do it')]}),
                'gen_ai.output.messages': _messages_attr(
                    {
                        'role': 'assistant',
                        'parts': [{'type': 'tool_call', 'id': 'call_1', 'name': 'search', 'arguments': {'q': 'x'}}],
                    }
                ),
            },
        ),
        _span(
            'tool1',
            'search',
            parent='c1',
            start=2_100_000,
            end=2_200_000,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('execute_tool')},
                {'key': 'gen_ai.tool.name', 'value': _str('search')},
                {'key': 'gen_ai.tool.call.id', 'value': _str('call_1')},
                {'key': 'gen_ai.tool.call.result', 'value': _str('found')},
            ],
        ),
    )

    trace, messages = import_otlp_trace(payload)
    tool_calls = [e for e in trace.events if e.type is EventType.TOOL_CALL]
    tool_results = [e for e in trace.events if e.type is EventType.TOOL_RESULT]

    # The tool_call part in messages emits TOOL_CALL on its own chat turn; the
    # tool span only adds TOOL_RESULT (no duplicate TOOL_CALL).
    assert len(tool_calls) == 1
    assert tool_calls[0].step == 0  # the chat turn that made the call
    assert tool_calls[0].payload['span_id'] == 'c1'
    assert tool_calls[0].payload['tool_call_id'] == 'call_1'
    assert tool_calls[0].payload['name'] == 'search'
    assert len(tool_results) == 1
    assert tool_results[0].step == 0  # same turn as c1
    assert tool_results[0].payload['tool_call_id'] == 'call_1'

    assistant = next(m for m in messages if isinstance(m, ChatMessageAssistant) and m.tool_calls)
    assert assistant.tool_calls[0].function.name == 'search'
    assert assistant.tool_calls[0].function.arguments == {'q': 'x'}


def test_tool_span_without_message_part_synthesises_tool_call():
    """A tool span with no matching message part must still produce TOOL_CALL."""
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2_000_000,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        ),
        _span(
            'tool1',
            'search',
            parent='c1',
            start=2_100_000,
            end=2_200_000,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('execute_tool')},
                {'key': 'gen_ai.tool.name', 'value': _str('search')},
                {'key': 'gen_ai.tool.call.id', 'value': _str('call_9')},
            ],
        ),
    )

    trace, _ = import_otlp_trace(payload)
    tool_calls = [e for e in trace.events if e.type is EventType.TOOL_CALL]
    assert len(tool_calls) == 1
    assert tool_calls[0].payload['tool_call_id'] == 'call_9'


def test_create_agent_does_not_open_a_run_boundary():
    payload = _export(
        _span('create', 'create_agent', start=0, end=1, attributes=[{'key': 'gen_ai.operation.name', 'value': _str('create_agent')}]),
        _span('invoke', 'invoke_agent', start=2, end=3, attributes=[{'key': 'gen_ai.operation.name', 'value': _str('invoke_agent')}]),
        _chat_span(
            'c1',
            parent='invoke',
            start=4,
            end=5,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        ),
    )

    trace, _ = import_otlp_trace(payload)
    run_starts = [e for e in trace.events if e.type is EventType.RUN_START]
    run_ends = [e for e in trace.events if e.type is EventType.RUN_END]

    assert len(run_starts) == 1 and len(run_ends) == 1
    assert run_starts[0].payload['span_id'] == 'invoke'


def test_valid_but_uninteresting_operation_is_skipped_and_counted():
    """embeddings is a legal operation; a RAG trace must not be marked as an error."""
    payload = _export(
        _span(
            'invoke',
            'invoke_agent',
            start=0,
            end=9,
            attributes=[{'key': 'gen_ai.operation.name', 'value': _str('invoke_agent')}],
        ),
        _chat_span(
            'c1',
            parent='invoke',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        ),
        _span('e1', 'embed', start=3, end=4, attributes=[{'key': 'gen_ai.operation.name', 'value': _str('embeddings')}]),
        _span('e2', 'embed', start=5, end=6, attributes=[{'key': 'gen_ai.operation.name', 'value': _str('embeddings')}]),
    )

    trace, _ = import_otlp_trace(payload)

    assert not [e for e in trace.events if e.type is EventType.ERROR]
    skipped = [e for e in trace.events if 'skipped_operations' in e.payload]
    assert skipped and skipped[0].payload['skipped_operations'] == {'embeddings': 2}


def test_span_error_status_becomes_error_event():
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        ),
        _span(
            'bad',
            'tool',
            start=3,
            end=4,
            attributes=[{'key': 'gen_ai.operation.name', 'value': _str('execute_tool')}],
            status={'code': 2, 'message': 'tool crashed'},
        ),
    )

    trace, _ = import_otlp_trace(payload)
    errors = [e for e in trace.events if e.type is EventType.ERROR]
    assert len(errors) == 1
    assert errors[0].payload['message'] == 'tool crashed'


def test_span_exception_event_becomes_error_event():
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        ),
        _span(
            'bad',
            'tool',
            start=3,
            end=4,
            attributes=[{'key': 'gen_ai.operation.name', 'value': _str('execute_tool')}],
            events=[{'name': 'exception', 'attributes': [{'key': 'exception.message', 'value': _str('boom')}]}],
        ),
    )

    trace, _ = import_otlp_trace(payload)
    errors = [e for e in trace.events if e.type is EventType.ERROR]
    assert errors and errors[0].payload['message'] == 'boom'


def test_legacy_flat_messages_are_accepted():
    """Pre-v1.36 exports use flat gen_ai.prompts / gen_ai.completions."""
    payload = _export(
        _span(
            'c1',
            'chat',
            start=1,
            end=2,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('chat')},
                {'key': 'gen_ai.usage.prompt_tokens', 'value': _int(3)},
                {'key': 'gen_ai.usage.completion_tokens', 'value': _int(4)},
                {'key': 'gen_ai.prompts', 'value': _str('legacy question')},
                {'key': 'gen_ai.completions', 'value': _str('legacy answer')},
            ],
        )
    )

    trace, messages = import_otlp_trace(payload)

    assert trace.events[0].token_usage == {'input': 3, 'output': 4, 'total': 7}
    assert messages[0].text == 'legacy question'
    assert messages[1].text == 'legacy answer'


def test_unknown_attribute_survives_in_payload():
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'vendor.custom.flag': _str('kept'),
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        )
    )

    trace, _ = import_otlp_trace(payload)
    assert trace.events[0].payload['vendor.custom.flag'] == 'kept'


def test_missing_message_content_is_rejected_with_content_capture_hint():
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
    def chat(span_id: str, sample_id: str, start: int) -> dict:
        return _chat_span(
            span_id,
            start=start,
            end=start + 1,
            **{
                'evalscope.sample_id': _str(sample_id),
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part(f'q-{sample_id}')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part(f'a-{sample_id}')]}),
            },
        )

    payload = _export(chat('s1', 'sample-a', 1), chat('s2', 'sample-b', 3))

    trace, messages = import_otlp_trace(payload, expected_sample_id='sample-b')

    assert len([e for e in trace.events if e.type is EventType.MODEL_GENERATE]) == 1
    assert messages[0].text == 'q-sample-b'


def test_unresolvable_sample_id_names_the_missing_value():
    payload = _export(
        _chat_span(
            's1',
            start=1,
            end=2,
            **{
                'evalscope.sample_id': _str('sample-a'),
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        )
    )

    with pytest.raises(OTLPImportError, match='sample-zzz'):
        import_otlp_trace(payload, expected_sample_id='sample-zzz')


def test_tool_result_message_is_reconstructed():
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr(
                    {'role': 'user', 'parts': [_text_part('search')]},
                    {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': 'c1', 'name': 'search', 'arguments': {}}]},
                    {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'c1', 'content': 'result'}]},
                ),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('done')]}),
            },
        )
    )

    _, messages = import_otlp_trace(payload)

    tool_messages = [m for m in messages if isinstance(m, ChatMessageTool)]
    assert len(tool_messages) == 1
    assert tool_messages[0].text == 'result'
    assert tool_messages[0].tool_call_id == 'c1'


# ---------------------------------------------------------------------------
# Regression tests for the five review findings on PR #2197
# ---------------------------------------------------------------------------


def test_legacy_span_does_not_duplicate_the_assistant_turn():
    """Legacy flat export: prompts -> user, completions -> assistant (once)."""
    payload = _export(
        _span(
            'c1',
            'chat',
            start=1,
            end=2,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('chat')},
                {'key': 'gen_ai.prompts', 'value': _str('legacy question')},
                {'key': 'gen_ai.completions', 'value': _str('legacy answer')},
            ],
        )
    )

    _, messages = import_otlp_trace(payload)

    assert [type(m).__name__ for m in messages] == ['ChatMessageUser', 'ChatMessageAssistant']
    assert [m.text for m in messages] == ['legacy question', 'legacy answer']


def test_later_chat_span_contributes_new_tool_observations():
    """gen_ai.input.messages re-sends history; only the new tail is appended."""
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2_000_000,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q1')]}),
                'gen_ai.output.messages': _messages_attr(
                    {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': 'call_1', 'name': 'search', 'arguments': {}}]}
                ),
            },
        ),
        _chat_span(
            'c2',
            start=3,
            end=4_000_000,
            **{
                # Full history re-sent, including the tool observation.
                'gen_ai.input.messages': _messages_attr(
                    {'role': 'user', 'parts': [_text_part('q1')]},
                    {
                        'role': 'assistant',
                        'parts': [{'type': 'tool_call', 'id': 'call_1', 'name': 'search', 'arguments': {}}],
                    },
                    {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'call_1', 'content': 'found'}]},
                ),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a2')]}),
            },
        ),
    )

    _, messages = import_otlp_trace(payload)
    texts = [getattr(m, 'text', None) for m in messages]

    # No duplicated 'q1'; the tool observation from the second input is kept.
    assert texts.count('q1') == 1
    assert 'found' in texts
    assert 'a2' in texts


def test_high_level_loader_rejects_trace_ending_on_a_tool_call(tmp_path):
    """A trace whose last assistant turn is a tool call has no final answer."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr(
                    {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': 'call_1', 'name': 'search', 'arguments': {}}]}
                ),
            },
        )
    )
    path = tmp_path / 'sample-1.otlp.json'
    path.write_text(_json(payload))

    with pytest.raises(OTLPImportError, match='ends on a tool call'):
        load_otlp_trace_for_sample(
            OTelTraceConfig(trace_path=str(tmp_path)),
            SimpleNamespace(name='model'),
            SimpleNamespace(id='sample-1'),
        )


def test_high_level_loader_filters_shared_export_by_sample_id(tmp_path):
    """A multi-sample export is filtered by evalscope.sample_id."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    def chat(span_id: str, sample_id: str, start: int) -> dict:
        return _chat_span(
            span_id,
            start=start,
            end=start + 1,
            **{
                'evalscope.sample_id': _str(sample_id),
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part(f'q-{sample_id}')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part(f'a-{sample_id}')]}),
            },
        )

    path = tmp_path / 'shared.otlp.json'
    path.write_text(_json(_export(chat('s1', 'sample-a', 1), chat('s2', 'sample-b', 3))))

    result = load_otlp_trace_for_sample(
        OTelTraceConfig(trace_path=str(path)),
        SimpleNamespace(name='model'),
        SimpleNamespace(id='sample-b'),
    )

    assert [m.text for m in result.messages] == ['q-sample-b', 'a-sample-b']
    assert result.output.completion == 'a-sample-b'


def test_default_adapter_routes_otel_trace_without_a_loop():
    """DefaultDataAdapter must not treat otel_trace as a native agent loop."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel_config import OTelTraceConfig
    from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter

    adapter = object.__new__(DefaultDataAdapter)
    adapter._task_config = SimpleNamespace(agent_config=OTelTraceConfig(trace_path='/nonexistent'))

    # No trace file -> the OTel loader raises OTLPImportError, not AttributeError
    # from dereferencing cfg.strategy on a native agent config.
    with pytest.raises(OTLPImportError):
        DefaultDataAdapter._on_inference(
            adapter, SimpleNamespace(name='model'), SimpleNamespace(id='sample-1')
        )


def _json(payload) -> str:
    import json

    return json.dumps(payload)


def test_high_level_loader_accepts_integer_sample_ids(tmp_path):
    """Production Sample.id is an int; OTLP attributes decode to str."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    def chat(span_id: str, sample_id: str, start: int) -> dict:
        return _chat_span(
            span_id,
            start=start,
            end=start + 1,
            **{
                'evalscope.sample_id': _str(sample_id),
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part(f'q-{sample_id}')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part(f'a-{sample_id}')]}),
            },
        )

    path = tmp_path / 'shared.otlp.json'
    path.write_text(_json(_export(chat('s1', '7', 1), chat('s2', '9', 3))))

    result = load_otlp_trace_for_sample(
        OTelTraceConfig(trace_path=str(path)),
        SimpleNamespace(name='model'),
        SimpleNamespace(id=9),  # int, as a production Sample carries
    )

    assert [m.text for m in result.messages] == ['q-9', 'a-9']
    assert result.output.completion == 'a-9'


def test_agent_identity_rides_on_run_start_payload():
    """gen_ai.agent.name/id must survive on the RUN_START payload.

    A create_agent span sets up the agent (no event); an invoke_agent span runs
    it. The identity may appear on either, and must not be filtered out.
    """
    payload = _export(
        _span(
            'create',
            'create_agent',
            start=0,
            end=1,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('create_agent')},
                {'key': 'gen_ai.agent.name', 'value': _str('planner')},
                {'key': 'gen_ai.agent.id', 'value': _str('agent-42')},
            ],
        ),
        _span('invoke', 'invoke_agent', start=2, end=3, attributes=[{'key': 'gen_ai.operation.name', 'value': _str('invoke_agent')}]),
        _chat_span(
            'c1',
            parent='invoke',
            start=4,
            end=5,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        ),
    )

    trace, _ = import_otlp_trace(payload)
    run_start = next(e for e in trace.events if e.type is EventType.RUN_START)

    assert run_start.payload['gen_ai.agent.name'] == 'planner'
    assert run_start.payload['gen_ai.agent.id'] == 'agent-42'


def test_invoke_agent_identity_wins_over_create_agent():
    """An identity on the invoke_agent span takes precedence over create_agent."""
    payload = _export(
        _span(
            'create',
            'create_agent',
            start=0,
            end=1,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('create_agent')},
                {'key': 'gen_ai.agent.name', 'value': _str('template')},
            ],
        ),
        _span(
            'invoke',
            'invoke_agent',
            start=2,
            end=3,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('invoke_agent')},
                {'key': 'gen_ai.agent.name', 'value': _str('instance')},
            ],
        ),
        _chat_span(
            'c1',
            parent='invoke',
            start=4,
            end=5,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        ),
    )

    trace, _ = import_otlp_trace(payload)
    run_start = next(e for e in trace.events if e.type is EventType.RUN_START)

    assert run_start.payload['gen_ai.agent.name'] == 'instance'


def test_later_legacy_chat_spans_keep_their_user_prompts():
    """Pre-v1.36 export with two chat spans must keep both user prompts."""
    payload = _export(
        _span(
            'c1',
            'chat',
            start=1,
            end=2,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('chat')},
                {'key': 'gen_ai.prompts', 'value': _str('q1')},
                {'key': 'gen_ai.completions', 'value': _str('a1')},
            ],
        ),
        _span(
            'c2',
            'chat',
            start=3,
            end=4,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('chat')},
                {'key': 'gen_ai.prompts', 'value': _str('q2')},
                {'key': 'gen_ai.completions', 'value': _str('a2')},
            ],
        ),
    )

    _, messages = import_otlp_trace(payload)
    texts = [getattr(m, 'text', None) for m in messages]

    assert texts == ['q1', 'a1', 'q2', 'a2']
    assert [type(m).__name__ for m in messages] == [
        'ChatMessageUser',
        'ChatMessageAssistant',
        'ChatMessageUser',
        'ChatMessageAssistant',
    ]


def test_integer_typed_sample_id_attributes_associate(tmp_path):
    """A natural integer evalscope.sample_id attribute must match sample.id."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    def chat(span_id: str, sample_id: int, start: int) -> dict:
        return _chat_span(
            span_id,
            start=start,
            end=start + 1,
            **{
                'evalscope.sample_id': _int(sample_id),  # intValue, not stringValue
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part(f'q-{sample_id}')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part(f'a-{sample_id}')]}),
            },
        )

    path = tmp_path / 'shared.otlp.json'
    path.write_text(_json(_export(chat('s1', 7, 1), chat('s2', 9, 3))))

    result = load_otlp_trace_for_sample(
        OTelTraceConfig(trace_path=str(path)),
        SimpleNamespace(name='model'),
        SimpleNamespace(id=9),
    )

    assert [m.text for m in result.messages] == ['q-9', 'a-9']


def test_skipped_operations_do_not_add_a_second_run_boundary():
    """Skipped-operation bookkeeping must reuse the existing RUN_END payload."""
    payload = _export(
        _span('invoke', 'invoke_agent', start=0, end=9, attributes=[{'key': 'gen_ai.operation.name', 'value': _str('invoke_agent')}]),
        _chat_span(
            'c1',
            parent='invoke',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        ),
        _span('e1', 'embed', start=3, end=4, attributes=[{'key': 'gen_ai.operation.name', 'value': _str('embeddings')}]),
    )

    trace, _ = import_otlp_trace(payload)
    run_ends = [e for e in trace.events if e.type is EventType.RUN_END]

    assert len(run_ends) == 1, 'skipped operations must not add a run boundary'
    assert run_ends[0].payload['skipped_operations'] == {'embeddings': 1}


def test_assistant_entry_with_text_and_tool_call_is_one_message():
    """[text, tool_call] parts must stay one assistant turn, not two."""
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr(
                    {
                        'role': 'assistant',
                        'parts': [
                            _text_part('Let me search'),
                            {'type': 'tool_call', 'id': 'call_1', 'name': 'search', 'arguments': {'q': 'x'}},
                        ],
                    }
                ),
            },
        )
    )

    _, messages = import_otlp_trace(payload)

    assert len(messages) == 2  # user + one assistant turn
    assistant = messages[1]
    assert isinstance(assistant, ChatMessageAssistant)
    assert assistant.text == 'Let me search'
    assert assistant.tool_calls and assistant.tool_calls[0].id == 'call_1'


def test_intermediate_text_with_tool_call_is_not_treated_as_missing_answer(tmp_path):
    """A turn with both text and a tool call can still be the prediction."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr(
                    {
                        'role': 'assistant',
                        'parts': [
                            _text_part('Working on it'),
                            {'type': 'tool_call', 'id': 'call_1', 'name': 'search', 'arguments': {}},
                        ],
                    }
                ),
            },
        )
    )
    (tmp_path / 'sample-1.otlp.json').write_text(_json(payload))

    result = load_otlp_trace_for_sample(
        OTelTraceConfig(trace_path=str(tmp_path)),
        SimpleNamespace(name='model'),
        SimpleNamespace(id='sample-1'),
    )

    assert result.output.completion == 'Working on it'


def test_sample_id_on_root_keeps_untagged_children():
    """A multi-sample export tags only the invoke_agent root; children inherit it."""
    payload = _export(
        _span(
            'root',
            'invoke_agent',
            start=0,
            end=9,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('invoke_agent')},
                {'key': 'evalscope.sample_id', 'value': _str('sample-a')},
            ],
        ),
        _chat_span(
            'c1',
            parent='root',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        ),
        _span(
            'other-root',
            'invoke_agent',
            start=10,
            end=19,
            attributes=[
                {'key': 'gen_ai.operation.name', 'value': _str('invoke_agent')},
                {'key': 'evalscope.sample_id', 'value': _str('sample-b')},
            ],
        ),
        _chat_span(
            'c2',
            parent='other-root',
            start=11,
            end=12,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q-b')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a-b')]}),
            },
        ),
    )

    trace, messages = import_otlp_trace(payload, expected_sample_id='sample-a')
    chat_spans = [e for e in trace.events if e.type is EventType.MODEL_GENERATE]

    # Both the tagged root and its untagged chat child belong to sample-a.
    assert len(chat_spans) == 1
    assert messages[0].text == 'q'


# ---------------------------------------------------------------------------
# Maintainer scope: one complete run per sample, evidence preserved
# ---------------------------------------------------------------------------


def test_tool_call_response_reads_the_standard_response_field():
    """GenAI semconv puts the tool result under ``response``, not ``content``."""
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr(
                    {'role': 'user', 'parts': [_text_part('search')]},
                    {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': 'c1', 'name': 'search', 'arguments': {}}]},
                    {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'c1', 'response': 'found'}]},
                ),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('done')]}),
            },
        )
    )

    _, messages = import_otlp_trace(payload)
    tool_message = next(m for m in messages if isinstance(m, ChatMessageTool))

    assert tool_message.text == 'found'
    assert tool_message.tool_call_id == 'c1'


def test_identical_tool_results_from_different_calls_are_kept():
    """Two different calls returning the same text must both survive."""
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr(
                    {'role': 'user', 'parts': [_text_part('q')]},
                    {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': 'call_1', 'name': 'ping', 'arguments': {}}]},
                    {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'call_1', 'response': 'ok'}]},
                    {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': 'call_2', 'name': 'ping', 'arguments': {}}]},
                    {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'call_2', 'response': 'ok'}]},
                ),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('done')]}),
            },
        )
    )

    _, messages = import_otlp_trace(payload)
    tool_messages = [m for m in messages if isinstance(m, ChatMessageTool)]

    assert [m.tool_call_id for m in tool_messages] == ['call_1', 'call_2']
    assert [m.text for m in tool_messages] == ['ok', 'ok']


def test_untagged_export_with_multiple_runs_is_rejected(tmp_path):
    """Two runs in one untagged file cannot be bound to a sample."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    def run(root: str, chat_id: str, start: int, question: str, answer: str) -> list:
        return [
            _span(
                root,
                'invoke_agent',
                start=start,
                end=start + 100,
                attributes=[{'key': 'gen_ai.operation.name', 'value': _str('invoke_agent')}],
            ),
            _chat_span(
                chat_id,
                parent=root,
                start=start + 1,
                end=start + 2,
                **{
                    'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part(question)]}),
                    'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part(answer)]}),
                },
            ),
        ]

    payload = _export(
        *run('r1', 'c1', 0, 'q1', 'a1'),
        *run('r2', 'c2', 200, 'q2', 'a2'),
    )
    path = tmp_path / 'untagged.otlp.json'
    path.write_text(_json(payload))

    with pytest.raises(OTLPImportError, match='2 runs'):
        load_otlp_trace_for_sample(
            OTelTraceConfig(trace_path=str(path)),
            SimpleNamespace(name='model'),
            SimpleNamespace(id=0),
        )


def test_tagged_export_with_multiple_runs_binds_by_sample_id(tmp_path):
    """A tagged export with two runs binds each sample to its own run."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    def run(root: str, chat_id: str, start: int, question: str, answer: str, sample_id: str) -> list:
        return [
            _span(
                root,
                'invoke_agent',
                start=start,
                end=start + 100,
                attributes=[
                    {'key': 'gen_ai.operation.name', 'value': _str('invoke_agent')},
                    {'key': 'evalscope.sample_id', 'value': _str(sample_id)},
                ],
            ),
            _chat_span(
                chat_id,
                parent=root,
                start=start + 1,
                end=start + 2,
                **{
                    'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part(question)]}),
                    'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part(answer)]}),
                },
            ),
        ]

    payload = _export(
        *run('r1', 'c1', 0, 'q1', 'a1', 'sample-a'),
        *run('r2', 'c2', 200, 'q2', 'a2', 'sample-b'),
    )
    path = tmp_path / 'tagged.otlp.json'
    path.write_text(_json(payload))

    first = load_otlp_trace_for_sample(
        OTelTraceConfig(trace_path=str(path)), SimpleNamespace(name='model'), SimpleNamespace(id='sample-a')
    )
    second = load_otlp_trace_for_sample(
        OTelTraceConfig(trace_path=str(path)), SimpleNamespace(name='model'), SimpleNamespace(id='sample-b')
    )

    assert first.output.completion == 'a1'
    assert second.output.completion == 'a2'


def test_sample_requiring_a_sandbox_is_rejected_before_scoring(tmp_path):
    """OTel import cannot provide a sandbox, so report before evaluation starts."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        )
    )
    path = tmp_path / 'sample-1.otlp.json'
    path.write_text(_json(payload))

    with pytest.raises(OTLPImportError, match='requires a sandbox'):
        load_otlp_trace_for_sample(
            OTelTraceConfig(trace_path=str(tmp_path)),
            SimpleNamespace(name='model'),
            SimpleNamespace(id='sample-1', sandbox={'type': 'docker'}),
        )


def test_otel_import_bypasses_an_on_inference_override():
    """run_inference intercepts before the polymorphic dispatch.

    Real agent benchmarks (cybench, bfcl, tau_bench, ...) override
    ``_on_inference``; the import must not depend on that hook.
    """
    from types import SimpleNamespace

    from evalscope.api.agent.otel_config import OTelTraceConfig
    from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter

    calls = []

    class _Override(DefaultDataAdapter):
        def _on_inference(self, model, sample):  # noqa: D102 - test double
            calls.append('dispatched')
            raise AssertionError('the import must run before _on_inference')

    adapter = object.__new__(_Override)
    adapter._task_config = SimpleNamespace(agent_config=OTelTraceConfig(trace_path='/nonexistent'))
    adapter._on_inference_start = lambda model, sample: calls.append('start')

    # The importer raises (missing file) instead of dispatching to _on_inference.
    with pytest.raises(OTLPImportError):
        DefaultDataAdapter.run_inference(adapter, SimpleNamespace(name='m'), SimpleNamespace(id=1), '/tmp')

    assert calls == ['start']


def test_tool_call_response_reads_the_result_field():
    """The OTel registry example spells the tool result ``result``."""
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr(
                    {'role': 'user', 'parts': [_text_part('weather')]},
                    {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': 'c1', 'name': 'weather', 'arguments': {}}]},
                    {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'c1', 'result': 'rainy, 57F'}]},
                ),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('done')]}),
            },
        )
    )

    _, messages = import_otlp_trace(payload)
    tool_message = next(m for m in messages if isinstance(m, ChatMessageTool))

    assert tool_message.text == 'rainy, 57F'


def test_turn_order_is_preserved():
    """The reconstructed message history keeps the user/assistant/tool order."""
    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr(
                    {'role': 'user', 'parts': [_text_part('q')]},
                    {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': 'c1', 'name': 'search', 'arguments': {}}]},
                    {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'c1', 'response': 'found'}]},
                ),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('answer')]}),
            },
        )
    )

    _, messages = import_otlp_trace(payload)

    assert [m.role for m in messages] == ['user', 'assistant', 'tool', 'assistant']


def test_consecutive_assistant_answers_are_rejected(tmp_path):
    """Two answers with no turn boundary cannot be resolved to a final one."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr(
                    {'role': 'assistant', 'parts': [_text_part('first')]},
                    {'role': 'assistant', 'parts': [_text_part('second')]},
                ),
            },
        )
    )
    path = tmp_path / 'sample-1.otlp.json'
    path.write_text(_json(payload))

    with pytest.raises(OTLPImportError, match='consecutive assistant answers'):
        load_otlp_trace_for_sample(
            OTelTraceConfig(trace_path=str(tmp_path)),
            SimpleNamespace(name='model'),
            SimpleNamespace(id='sample-1'),
        )


def test_single_untagged_export_cannot_serve_multiple_samples(tmp_path):
    """One untagged run must not be reused as every sample's answer."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('first question')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('first answer')]}),
            },
        )
    )
    path = tmp_path / 'shared.otlp.json'
    path.write_text(_json(payload))

    with pytest.raises(OTLPImportError, match='cannot serve 2 samples'):
        load_otlp_trace_for_sample(
            OTelTraceConfig(trace_path=str(path)),
            SimpleNamespace(name='model'),
            SimpleNamespace(id=1),
            n_samples=2,
        )


def test_single_untagged_export_serves_one_sample(tmp_path):
    """A single-sample dataset may use one untagged export."""
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    payload = _export(
        _chat_span(
            'c1',
            start=1,
            end=2,
            **{
                'gen_ai.input.messages': _messages_attr({'role': 'user', 'parts': [_text_part('q')]}),
                'gen_ai.output.messages': _messages_attr({'role': 'assistant', 'parts': [_text_part('a')]}),
            },
        )
    )
    path = tmp_path / 'only.otlp.json'
    path.write_text(_json(payload))

    result = load_otlp_trace_for_sample(
        OTelTraceConfig(trace_path=str(path)),
        SimpleNamespace(name='model'),
        SimpleNamespace(id=0),
        n_samples=1,
    )
    assert result.output.completion == 'a'


def test_exporter_fixture_scores_known_answer_and_tool_result(tmp_path):
    """End-to-end: an exporter-produced fixture scores its recorded run.

    The two files under ``fixtures/otel`` were generated by the OpenTelemetry
    Python SDK (see ``fixtures/otel/_generate.py``), so this exercises the
    real OTLP/JSON wire shape, not a hand-rolled dict. The scorer reads the
    reconstructed tool message: if the tool result were dropped, the answer
    would still match the text but ``reference in tool_results`` would fail, so
    the 2/2 score proves the tool result reached the scorer.
    """
    from pathlib import Path

    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.api.benchmark.adapters.agent_adapter import AgentAdapter
    from evalscope.api.dataset import DatasetDict, Sample
    from evalscope.api.metric import Score
    from evalscope.api.registry import BENCHMARK_REGISTRY, register_benchmark
    from evalscope.config import TaskConfig
    from evalscope.run import run_task

    fixture_dir = Path(__file__).parent / 'fixtures' / 'otel'
    name = 'otel_fixture_e2e'

    @register_benchmark(
        BenchmarkMeta(
            name=name,
            pretty_name='OTel Fixture E2E',
            dataset_id='local',
            description='Scores an exporter-produced OTLP trace against its sample.',
            subset_list=['default'],
            metric_list=['acc'],
            eval_split='test',
            prompt_template='{question}',
        )
    )
    class _OtelFixtureAdapter(AgentAdapter):

        def load(self):
            return DatasetDict(
                {
                    'default': [
                        Sample(id=0, input='What is the weather in Paris?', target='rainy'),
                        Sample(id=1, input='What is the capital of France?', target='Paris'),
                    ]
                }
            ), None

        def match_score(self, original_prediction, filtered_prediction, reference, task_state):
            tool_results = [m.text for m in task_state.messages if isinstance(m, ChatMessageTool)]
            correct = reference in (filtered_prediction or '') and reference in tool_results
            return Score(
                extracted_prediction=filtered_prediction,
                prediction=original_prediction,
                value={'acc': 1.0 if correct else 0.0},
            )

    try:
        scores = run_task(
            TaskConfig(
                model='mock',
                datasets=[name],
                agent_config={'mode': 'otel_trace', 'trace_path': str(fixture_dir)},
                eval_type='mock_llm',
                work_dir=str(tmp_path / 'out'),
                limit=2,
            )
        )
    finally:
        BENCHMARK_REGISTRY.pop(name, None)

    metric = scores[name].metrics[0]
    assert metric.num == 2
    assert metric.num_correct == 2
