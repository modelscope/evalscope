"""Import OpenTelemetry GenAI traces into EvalScope's ``AgentTrace``.

Production agents increasingly emit OTel traces following the GenAI semantic
conventions (``gen_ai.operation.name``, ``gen_ai.usage.*``,
``gen_ai.input.messages`` / ``gen_ai.output.messages``). This module turns an
OTLP JSON export into the same ``AgentTrace`` + ``ChatMessage`` pair the
external-agent bridge produces, so an existing trace becomes a scorable sample
without re-running the agent under EvalScope.

Design notes:

* **Third producer, not a new mode.** The result feeds the existing
  ``_maybe_run_external_agent`` extension point; ``AgentTrace.framework`` is a
  free-form string, so ``'otel-import'`` needs no contract change.
* **Fail loudly.** GenAI semantic conventions make message content opt-in, so
  an export can carry metadata only. Such an export cannot reconstruct the
  trajectory and is rejected with the offending span id rather than imported as
  an empty trace that silently scores 0.
* **Unknown attributes survive.** Vendor-specific and future-semconv attributes
  are preserved in ``AgentTraceEvent.payload`` so the importer does not need a
  release per OTel version.
"""

from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.agent.trace import AgentTrace, AgentTraceEvent, EventType
from evalscope.api.messages import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from evalscope.api.tool import ToolCall, ToolFunction

__all__ = ['OTLPImportError', 'import_otlp_trace']

#: ``AgentTrace.framework`` value for imported traces (free-form string).
FRAMEWORK = 'otel-import'

#: Attribute the importer reads for sample association when several samples
#: share one export. Absent means "one export per sample" (the default).
DEFAULT_SAMPLE_ID_ATTRIBUTE = 'evalscope.sample_id'

# GenAI semantic-convention attribute names (2026-11 baseline; pre-2025
# names are accepted through ``_USAGE_FALLBACK``).
_OP_NAME = 'gen_ai.operation.name'
_USAGE_INPUT = 'gen_ai.usage.input_tokens'
_USAGE_OUTPUT = 'gen_ai.usage.output_tokens'
_USAGE_FALLBACK = {
    'gen_ai.usage.prompt_tokens': _USAGE_INPUT,
    'gen_ai.usage.completion_tokens': _USAGE_OUTPUT,
}
_INPUT_MESSAGES = 'gen_ai.input.messages'
_OUTPUT_MESSAGES = 'gen_ai.output.messages'

#: ``gen_ai.operation.name`` values this importer understands.
_OP_CHAT = 'chat'
_OP_TEXT_COMPLETION = 'text_completion'
_OP_EXECUTE_TOOL = 'execute_tool'
_OP_INVOKE_AGENT = 'invoke_agent'
_OP_CREATE_AGENT = 'create_agent'


class OTLPImportError(ValueError):
    """An OTLP export cannot be mapped to a scorable sample.

    Every instance names the offending span so the user can find it in the
    export; the importer never degrades to an empty trace.
    """


def _attr_value(value: Dict[str, Any]) -> Any:
    """Decode one OTel ``AnyValue`` into a Python value."""
    if 'stringValue' in value:
        return value['stringValue']
    if 'intValue' in value:
        return int(value['intValue'])
    if 'doubleValue' in value:
        return float(value['doubleValue'])
    if 'boolValue' in value:
        return bool(value['boolValue'])
    if 'bytesValue' in value:
        return value['bytesValue']
    if 'arrayValue' in value:
        return [_attr_value(item) for item in value['arrayValue'].get('values', [])]
    if 'kvlistValue' in value:
        return {item['key']: _attr_value(item['value']) for item in value['kvlistValue'].get('values', [])}
    return None


def _span_attributes(span: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a span's attribute list, applying the pre-2025 usage aliases."""
    attrs: Dict[str, Any] = {}
    for item in span.get('attributes', []):
        key = item.get('key')
        if not key:
            continue
        attrs[_USAGE_FALLBACK.get(key, key)] = _attr_value(item.get('value', {}))
    return attrs


def _span_id(span: Dict[str, Any]) -> str:
    return span.get('spanId') or span.get('name') or '<unknown span>'


def _iter_spans(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten ``resourceSpans`` → ``scopeSpans`` → ``spans``."""
    spans: List[Dict[str, Any]] = []
    for resource in payload.get('resourceSpans', []):
        for scope in resource.get('scopeSpans', []):
            spans.extend(scope.get('spans', []))
    return spans


def _start_ns(span: Dict[str, Any]) -> int:
    try:
        return int(span.get('startTimeUnixNano') or 0)
    except (TypeError, ValueError):
        return 0


def _step_depths(spans: List[Dict[str, Any]]) -> Dict[str, int]:
    """Map span id → nesting depth (root spans are 0)."""
    by_id = {_span_id(span): span for span in spans}
    depths: Dict[str, int] = {}

    def depth(span_id: str, seen: Optional[set] = None) -> int:
        if span_id in depths:
            return depths[span_id]
        seen = seen or set()
        if span_id in seen:  # cycle guard — malformed export
            return 0
        seen.add(span_id)
        span = by_id.get(span_id)
        parent = (span or {}).get('parentSpanId')
        value = 0 if not parent or parent not in by_id else depth(parent, seen) + 1
        depths[span_id] = value
        return value

    for span in spans:
        depth(_span_id(span))
    return depths


def _usage(attrs: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Build ``token_usage`` from ``gen_ai.usage.*`` when both counts exist."""
    prompt = attrs.get(_USAGE_INPUT)
    completion = attrs.get(_USAGE_OUTPUT)
    if prompt is None and completion is None:
        return None
    usage = {
        'input': int(prompt or 0),
        'output': int(completion or 0),
    }
    usage['total'] = usage['input'] + usage['output']
    return usage


def _part_to_message(role: str, part: Dict[str, Any], model: Optional[str] = None) -> Optional[ChatMessage]:
    """Convert one ``gen_ai.*.messages[].parts[]`` entry into a ChatMessage.

    The ``parts[].type`` discriminator follows the GenAI semantic conventions:
    ``text`` for content, ``tool_call`` for a function call, and
    ``tool_call_response`` for a tool result.
    """
    part_type = part.get('type', 'text')
    if part_type == 'text':
        content = part.get('content', '')
        if role == 'user':
            return ChatMessageUser(content=content)
        if role == 'system':
            return ChatMessageSystem(content=content)
        if role == 'tool':
            return ChatMessageTool(content=content, tool_call_id=part.get('id'))
        return ChatMessageAssistant(content=content, model=model)
    if part_type == 'tool_call':
        # ``ToolFunction.arguments`` is a dict; its validator parses JSON strings too.
        arguments = part.get('arguments')
        if not isinstance(arguments, (dict, str)):
            arguments = _json_dumps(arguments)
        return ChatMessageAssistant(
            content='',
            model=model,
            tool_calls=[
                ToolCall(
                    id=str(part.get('id') or ''),
                    function=ToolFunction(name=str(part.get('name') or ''), arguments=arguments),
                )
            ],
        )
    if part_type == 'tool_call_response':
        content = part.get('content')
        if not isinstance(content, str):
            content = _json_dumps(content)
        return ChatMessageTool(content=content, tool_call_id=part.get('id'))
    # Unknown part type: keep it visible rather than dropping content silently.
    return ChatMessageSystem(content=_json_dumps({'type': part_type, 'raw': part}))


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


def _messages_from_attr(value: Any, model: Optional[str]) -> List[ChatMessage]:
    """Expand one ``gen_ai.*.messages`` attribute into ChatMessage objects."""
    if not isinstance(value, list):
        return []
    messages: List[ChatMessage] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        role = entry.get('role', 'user')
        parts = entry.get('parts')
        if not isinstance(parts, list) or not parts:
            # Some instrumentations put a plain ``content`` on the entry.
            content = entry.get('content')
            if content is None:
                continue
            parts = [{'type': 'text', 'content': content}]
        for part in parts:
            message = _part_to_message(role, part, model=model)
            if message is not None:
                messages.append(message)
    return messages


def _merge_assistant_tool_calls(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Fold consecutive assistant tool-call messages into one assistant turn.

    ``parts[]`` produces one message per part; a single assistant turn with two
    tool calls should stay one assistant message with two ``tool_calls``.
    """
    merged: List[ChatMessage] = []
    for message in messages:
        last = merged[-1] if merged else None
        if (
            isinstance(message, ChatMessageAssistant)
            and isinstance(last, ChatMessageAssistant)
            and last.tool_calls
            and message.tool_calls
        ):
            merged[-1] = last.model_copy(update={'tool_calls': list(last.tool_calls) + list(message.tool_calls)})
            continue
        merged.append(message)
    return merged


def import_otlp_trace(
    payload: Dict[str, Any],
    *,
    sample_id_attribute: str = DEFAULT_SAMPLE_ID_ATTRIBUTE,
    expected_sample_id: Optional[str] = None,
) -> Tuple[AgentTrace, List[ChatMessage]]:
    """Convert an OTLP JSON export into ``(AgentTrace, messages)``.

    Args:
        payload: The parsed OTLP JSON export (``resourceSpans`` →
            ``scopeSpans`` → ``spans``).
        sample_id_attribute: Attribute carrying the eval sample id when several
            samples share one export.
        expected_sample_id: When set, the export must contain a span whose
            ``sample_id_attribute`` equals this value (one export per sample
            keeps the id optional; multi-sample exports require it).

    Returns:
        The ``AgentTrace`` (``framework='otel-import'``) and the reconstructed
        message history, in the shape ``InferenceResult`` expects.

    Raises:
        OTLPImportError: The export is empty, has no GenAI operation spans, is
            missing message content, or cannot be associated with a sample.
    """
    spans = _iter_spans(payload)
    if not spans:
        raise OTLPImportError('OTLP export contains no spans')

    # Sample association: a declared attribute wins; otherwise one export per sample.
    if expected_sample_id is not None:
        matched = [span for span in spans if _span_attributes(span).get(sample_id_attribute) == expected_sample_id]
        if not matched:
            raise OTLPImportError(
                f'No span carries {sample_id_attribute}={expected_sample_id!r}; '
                f'found sample ids: {sorted({_span_attributes(s).get(sample_id_attribute) for s in spans})}'
            )
        spans = matched

    depths = _step_depths(spans)
    ordered = sorted(spans, key=_start_ns)

    trace = AgentTrace(framework=FRAMEWORK)
    messages: List[ChatMessage] = []
    chat_turns = 0
    saw_content = False
    tool_results: Dict[str, ChatMessage] = {}

    for span in ordered:
        span_id = _span_id(span)
        attrs = _span_attributes(span)
        operation = attrs.get(_OP_NAME)
        if operation is None:
            raise OTLPImportError(f'Span {span_id!r} has no {_OP_NAME!r}; only GenAI operation spans can be imported')
        depth = depths.get(span_id, 0)
        model = attrs.get('gen_ai.request.model') or attrs.get('gen_ai.response.model')
        payload_extra = {k: v for k, v in attrs.items() if k not in _KNOWN_ATTRIBUTES}
        if payload_extra:
            # Unknown/vendor attributes survive instead of being dropped.
            pass

        if operation in (_OP_CHAT, _OP_TEXT_COMPLETION):
            turn = chat_turns
            chat_turns += 1
            input_messages = _messages_from_attr(attrs.get(_INPUT_MESSAGES), model)
            if turn == 0 and input_messages:
                messages.extend(input_messages)
                saw_content = True
            output_messages = _messages_from_attr(attrs.get(_OUTPUT_MESSAGES), model)
            if output_messages:
                messages.extend(output_messages)
                saw_content = True
            trace.add_event(
                step=turn,
                type=EventType.MODEL_GENERATE,
                latency_ms=_latency_ms(span),
                token_usage=_usage(attrs),
                payload={'span_id': span_id, 'operation': operation, **payload_extra},
            )
            continue

        if operation == _OP_EXECUTE_TOOL:
            tool_name = attrs.get('gen_ai.tool.name') or span.get('name')
            tool_args = attrs.get('gen_ai.tool.call.arguments')
            tool_result = attrs.get('gen_ai.tool.call.result')
            call_id = str(attrs.get('gen_ai.tool.call.id') or span_id)
            trace.add_event(
                step=depth,
                type=EventType.TOOL_CALL,
                latency_ms=_latency_ms(span),
                payload={
                    'span_id': span_id,
                    'operation': operation,
                    'tool_call_id': call_id,
                    'name': tool_name,
                    'arguments': tool_args,
                    **payload_extra,
                },
            )
            if tool_result is not None:
                tool_results[call_id] = ChatMessageTool(
                    content=tool_result if isinstance(tool_result, str) else _json_dumps(tool_result),
                    tool_call_id=call_id,
                )
            trace.add_event(
                step=depth,
                type=EventType.TOOL_RESULT,
                payload={'span_id': span_id, 'operation': operation, 'tool_call_id': call_id, **payload_extra},
            )
            continue

        if operation in (_OP_INVOKE_AGENT, _OP_CREATE_AGENT):
            trace.add_event(
                step=depth,
                type=EventType.RUN_START,
                payload={'span_id': span_id, 'operation': operation, **payload_extra},
            )
            trace.add_event(
                step=depth,
                type=EventType.RUN_END,
                latency_ms=_latency_ms(span),
                payload={'span_id': span_id, 'operation': operation, **payload_extra},
            )
            continue

        # Unknown operation name: keep it visible, do not invent an EventType.
        trace.add_event(
            step=depth,
            type=EventType.ERROR,
            payload={'span_id': span_id, 'operation': operation, 'unsupported_operation': True, **payload_extra},
        )

    if chat_turns == 0:
        raise OTLPImportError(
            f'No {_OP_CHAT!r} spans found; cannot reconstruct a message history '
            f'(operations seen: {sorted({_span_attributes(s).get(_OP_NAME) for s in spans})})'
        )
    if not saw_content:
        raise OTLPImportError(
            f'No message content found in {chat_turns} chat span(s). GenAI semantic conventions make '
            f'content capture opt-in; enable it on the exporter so {_INPUT_MESSAGES!r} / '
            f'{_OUTPUT_MESSAGES!r} are emitted'
        )

    return trace, _merge_assistant_tool_calls(messages)


def _latency_ms(span: Dict[str, Any]) -> Optional[float]:
    """Span duration in milliseconds, when start/end timestamps are present."""
    try:
        start = int(span.get('startTimeUnixNano') or 0)
        end = int(span.get('endTimeUnixNano') or 0)
    except (TypeError, ValueError):
        return None
    if not start or not end or end < start:
        return None
    return (end - start) / 1e6


#: Attributes read directly by the importer; everything else is preserved verbatim
#: in the event payload.
_KNOWN_ATTRIBUTES = {
    _OP_NAME,
    _USAGE_INPUT,
    _USAGE_OUTPUT,
    *_USAGE_FALLBACK,
    _INPUT_MESSAGES,
    _OUTPUT_MESSAGES,
    'gen_ai.request.model',
    'gen_ai.response.model',
    'gen_ai.tool.name',
    'gen_ai.tool.call.arguments',
    'gen_ai.tool.call.result',
    'gen_ai.tool.call.id',
}
