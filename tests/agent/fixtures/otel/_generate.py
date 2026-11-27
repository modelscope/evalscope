"""Regenerate the OTLP JSON fixtures used by the importer's end-to-end test.

The fixtures are **not** hand-written OTLP JSON: real ``ReadableSpan`` objects
are produced by the OpenTelemetry Python SDK's ``TracerProvider`` (the same
provider ``opentelemetry-instrumentation-*`` installs in an application) and
then serialised with the OTLP/JSON encoding that
``opentelemetry-exporter-otlp-proto-json`` emits (camelCase keys,
``startTimeUnixNano`` as a decimal string, ``AnyValue`` unions, hex ids).

Run it from the repository root:

    python tests/agent/fixtures/otel/_generate.py

Requires the ``opentelemetry-sdk`` package. The generated JSON is committed,
so running the test suite does **not** need the SDK, and ``pyproject.toml`` is
unchanged (no new runtime dependency).

Timestamps are pinned so re-running the script produces a stable diff; the
trace and span ids are random and are frozen once the files are committed.
"""
import json
import os
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind

HERE = os.path.dirname(os.path.abspath(__file__))

# OTLP/JSON encodes the protobuf SpanKind enum, which is offset by one from the
# Python ``SpanKind`` enum (Python starts at 0, protobuf SPAN_KIND_INTERNAL=1).
_SPAN_KIND = {
    SpanKind.INTERNAL: 1,
    SpanKind.SERVER: 2,
    SpanKind.CLIENT: 3,
    SpanKind.PRODUCER: 4,
    SpanKind.CONSUMER: 5,
}


def _any_value(value: Any) -> Dict[str, Any]:
    """Encode a Python value as an OTLP/JSON ``AnyValue``."""
    if isinstance(value, bool):
        return {'boolValue': value}
    if isinstance(value, int):
        # int64 is a decimal string in OTLP/JSON.
        return {'intValue': str(value)}
    if isinstance(value, float):
        return {'doubleValue': value}
    if isinstance(value, (list, tuple)):
        return {'arrayValue': {'values': [_any_value(item) for item in value]}}
    if isinstance(value, dict):
        return {'kvlistValue': {'values': [{'key': key, 'value': _any_value(item)} for key, item in value.items()]}}
    return {'stringValue': str(value)}


def _key_values(attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{'key': key, 'value': _any_value(value)} for key, value in attributes.items()]


def _span_to_otlp(span: ReadableSpan) -> Dict[str, Any]:
    """Serialise one ``ReadableSpan`` into an OTLP/JSON span object."""
    context = span.get_span_context()
    parent = span.parent
    status = span.status
    encoded: Dict[str, Any] = {
        'traceId': format(context.trace_id, '032x'),
        'spanId': format(context.span_id, '016x'),
        'name': span.name,
        'kind': _SPAN_KIND.get(span.kind, 1),
        'startTimeUnixNano': str(span.start_time),
        'endTimeUnixNano': str(span.end_time),
        'attributes': _key_values(dict(span.attributes)),
        'status': {
            'code': status.status_code.value,
            **({'message': status.description} if status.description else {}),
        },
    }
    if parent is not None:
        encoded['parentSpanId'] = format(parent.span_id, '016x')
    if span.events:
        encoded['events'] = [
            {
                'name': event.name,
                'timeUnixNano': str(event.timestamp),
                'attributes': _key_values(dict(event.attributes or {})),
            }
            for event in span.events
        ]
    return encoded


class _OtlpJsonExporter(SpanExporter):
    """Capture spans and render the OTLP/JSON ``resourceSpans`` envelope."""

    def __init__(self) -> None:
        self.spans: List[ReadableSpan] = []

    def export(self, spans) -> SpanExportResult:
        # ``SimpleSpanProcessor`` exports one span at a time; accumulate them so
        # a run lands in a single ``resourceSpans`` envelope.
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def envelope(self) -> Optional[Dict[str, Any]]:
        if not self.spans:
            return None
        resource = self.spans[0].resource
        scope = self.spans[0].instrumentation_scope
        scope_json: Dict[str, Any] = {'name': scope.name if scope else ''}
        if scope is not None and scope.version:
            scope_json['version'] = scope.version
        return {
            'resourceSpans': [
                {
                    'resource': {'attributes': _key_values(dict(resource.attributes))},
                    'scopeSpans': [
                        {
                            'scope': scope_json,
                            'spans': [_span_to_otlp(span) for span in self.spans],
                        }
                    ],
                }
            ]
        }


def _user(text: str) -> Dict[str, Any]:
    return {'role': 'user', 'parts': [{'type': 'text', 'content': text}]}


def _assistant(text: str) -> Dict[str, Any]:
    return {'role': 'assistant', 'parts': [{'type': 'text', 'content': text}]}


def _tool_call_part(call_id: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {'role': 'assistant', 'parts': [{'type': 'tool_call', 'id': call_id, 'name': name, 'arguments': arguments}]}


def _tool_response_part(call_id: str, response: str) -> Dict[str, Any]:
    return {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': call_id, 'response': response}]}


def _record_weather_run(tracer: trace.Tracer) -> None:
    """One run: ask for the weather, call a tool, answer from the tool result."""
    with tracer.start_as_current_span('invoke_agent', kind=SpanKind.INTERNAL) as run:
        run.set_attribute('gen_ai.operation.name', 'invoke_agent')
        run.set_attribute('gen_ai.agent.name', 'weather_agent')

        with tracer.start_as_current_span('chat', kind=SpanKind.CLIENT) as chat:
            chat.set_attribute('gen_ai.operation.name', 'chat')
            chat.set_attribute('gen_ai.request.model', 'gpt-demo')
            chat.set_attribute('gen_ai.usage.input_tokens', 18)
            chat.set_attribute('gen_ai.usage.output_tokens', 12)
            chat.set_attribute('gen_ai.input.messages', [_user('What is the weather in Paris?')])
            chat.set_attribute(
                'gen_ai.output.messages',
                [_tool_call_part('call_weather_1', 'get_weather', {'city': 'Paris'})],
            )

        with tracer.start_as_current_span('execute_tool get_weather', kind=SpanKind.INTERNAL) as tool:
            tool.set_attribute('gen_ai.operation.name', 'execute_tool')
            tool.set_attribute('gen_ai.tool.name', 'get_weather')
            tool.set_attribute('gen_ai.tool.call.id', 'call_weather_1')
            tool.set_attribute('gen_ai.tool.call.result', 'rainy')

        with tracer.start_as_current_span('chat', kind=SpanKind.CLIENT) as chat:
            chat.set_attribute('gen_ai.operation.name', 'chat')
            chat.set_attribute('gen_ai.request.model', 'gpt-demo')
            chat.set_attribute('gen_ai.usage.input_tokens', 31)
            chat.set_attribute('gen_ai.usage.output_tokens', 9)
            chat.set_attribute(
                'gen_ai.input.messages',
                [_user('What is the weather in Paris?'), _tool_response_part('call_weather_1', 'rainy')],
            )
            chat.set_attribute('gen_ai.output.messages', [_assistant('The weather in Paris is rainy.')])


def _record_capital_run(tracer: trace.Tracer) -> None:
    """One run: ask for the capital, call a tool, answer from the tool result."""
    with tracer.start_as_current_span('invoke_agent', kind=SpanKind.INTERNAL) as run:
        run.set_attribute('gen_ai.operation.name', 'invoke_agent')
        run.set_attribute('gen_ai.agent.name', 'geo_agent')

        with tracer.start_as_current_span('chat', kind=SpanKind.CLIENT) as chat:
            chat.set_attribute('gen_ai.operation.name', 'chat')
            chat.set_attribute('gen_ai.request.model', 'gpt-demo')
            chat.set_attribute('gen_ai.usage.input_tokens', 17)
            chat.set_attribute('gen_ai.usage.output_tokens', 11)
            chat.set_attribute('gen_ai.input.messages', [_user('What is the capital of France?')])
            chat.set_attribute(
                'gen_ai.output.messages',
                [_tool_call_part('call_capital_1', 'lookup', {'query': 'capital of France'})],
            )

        with tracer.start_as_current_span('execute_tool lookup', kind=SpanKind.INTERNAL) as tool:
            tool.set_attribute('gen_ai.operation.name', 'execute_tool')
            tool.set_attribute('gen_ai.tool.name', 'lookup')
            tool.set_attribute('gen_ai.tool.call.id', 'call_capital_1')
            tool.set_attribute('gen_ai.tool.call.result', 'Paris')

        with tracer.start_as_current_span('chat', kind=SpanKind.CLIENT) as chat:
            chat.set_attribute('gen_ai.operation.name', 'chat')
            chat.set_attribute('gen_ai.request.model', 'gpt-demo')
            chat.set_attribute('gen_ai.usage.input_tokens', 30)
            chat.set_attribute('gen_ai.usage.output_tokens', 8)
            chat.set_attribute(
                'gen_ai.input.messages',
                [_user('What is the capital of France?'), _tool_response_part('call_capital_1', 'Paris')],
            )
            chat.set_attribute('gen_ai.output.messages', [_assistant('The capital of France is Paris.')])


def build_payload(recorder) -> Dict[str, Any]:
    """Run one recorder through a real ``TracerProvider`` and return the export."""
    exporter = _OtlpJsonExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('evalscope.otel.fixture')
    recorder(tracer)
    provider.shutdown()
    payload = exporter.envelope()
    assert payload is not None, 'no spans captured'
    return payload


def _record(name: str, recorder) -> None:
    payload = build_payload(recorder)
    path = os.path.join(HERE, name)
    with open(path, 'w', encoding='utf-8') as handle:
        # Compact (single line): trace/span ids and timestamps are random per
        # run, so a pretty-printed diff is noise. The structure-drift test
        # compares span names / attribute keys, not these ids.
        json.dump(payload, handle, separators=(',', ':'), ensure_ascii=False)
        handle.write('\n')
    print(f'wrote {path}')


RUNS = {'0.otlp.json': _record_weather_run, '1.otlp.json': _record_capital_run}


def main() -> None:
    for name, recorder in RUNS.items():
        _record(name, recorder)


if __name__ == '__main__':
    main()
