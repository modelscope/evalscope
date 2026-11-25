"""Import OpenTelemetry GenAI traces into EvalScope's ``AgentTrace``.

Production agents increasingly emit OTel traces following the GenAI semantic
conventions (``gen_ai.operation.name``, ``gen_ai.usage.*``,
``gen_ai.input.messages`` / ``gen_ai.output.messages``). This module turns an
OTLP JSON export into the same ``AgentTrace`` + ``ChatMessage`` pair the
external-agent bridge produces, so an existing trace becomes a scorable sample
without re-running the agent under EvalScope.

Design notes:

* **Third producer, not a new mode.** The result feeds the existing
  ``_maybe_import_otel_trace`` extension point; ``AgentTrace.framework`` is a
  free-form string, so ``'otel-import'`` needs no ``EventType`` /
  ``AgentTrace`` / ``ReviewResult`` change (``TaskConfig`` gains a third
  ``agent_config`` mode, which is the public contract for that field).
* **Fail loudly.** GenAI semantic conventions make message content opt-in, so
  an export can carry metadata only. Such an export cannot reconstruct the
  trajectory and is rejected with the offending span id rather than imported as
  an empty trace that silently scores 0. A final assistant turn that carries a
  tool call and no text is an error, not an empty prediction.
* **Unknown attributes survive.** Vendor-specific and future-semconv attributes
  are preserved in ``AgentTraceEvent.payload`` so the importer does not need a
  release per OTel version.

Semantic-convention baseline: GenAI conventions are still at Development status
(https://opentelemetry.io/docs/specs/semconv/gen-ai/). The importer is written
against attribute names that have been stable across the v1.36-v1.43 line, and
accepts the pre-v1.36 flat names (``gen_ai.prompts`` / ``gen_ai.completions``
and ``gen_ai.usage.prompt_tokens`` / ``completion_tokens``) as a fallback.
"""

from collections import Counter
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

# GenAI semantic-convention attribute names (stable across v1.36-v1.43).
_OP_NAME = 'gen_ai.operation.name'
_USAGE_INPUT = 'gen_ai.usage.input_tokens'
_USAGE_OUTPUT = 'gen_ai.usage.output_tokens'
_INPUT_MESSAGES = 'gen_ai.input.messages'
_OUTPUT_MESSAGES = 'gen_ai.output.messages'

#: Pre-v1.36 flat names, still emitted by default by existing instrumentations.
_LEGACY_PROMPTS = 'gen_ai.prompts'
_LEGACY_COMPLETIONS = 'gen_ai.completions'
_LEGACY_ALIASES = {
    'gen_ai.usage.prompt_tokens': _USAGE_INPUT,
    'gen_ai.usage.completion_tokens': _USAGE_OUTPUT,
}

#: ``gen_ai.operation.name`` values this importer maps onto events.
_OP_CHAT = 'chat'
_OP_TEXT_COMPLETION = 'text_completion'
_OP_EXECUTE_TOOL = 'execute_tool'
_OP_INVOKE_AGENT = 'invoke_agent'
_OP_CREATE_AGENT = 'create_agent'

#: Attributes read directly by the importer; everything else is preserved
#: verbatim in the event payload.
_KNOWN_ATTRIBUTES = {
    _OP_NAME,
    _USAGE_INPUT,
    _USAGE_OUTPUT,
    *_LEGACY_ALIASES,
    _INPUT_MESSAGES,
    _OUTPUT_MESSAGES,
    _LEGACY_PROMPTS,
    _LEGACY_COMPLETIONS,
    'gen_ai.request.model',
    'gen_ai.response.model',
    'gen_ai.tool.name',
    'gen_ai.tool.call.arguments',
    'gen_ai.tool.call.result',
    'gen_ai.tool.call.id',
}

#: Attributes naming the agent that produced the trace; carried on RUN_START.
_AGENT_IDENTITY_KEYS = ('gen_ai.agent.name', 'gen_ai.agent.id')

#: OTel span status codes (``status.code``).
_STATUS_ERROR = 2


class OTLPImportError(ValueError):
    """An OTLP export cannot be mapped to a scorable sample.

    Every instance names the offending span so the user can find it in the
    export; the importer never degrades to an empty trace.
    """


# ---------------------------------------------------------------------------
# OTLP decoding helpers
# ---------------------------------------------------------------------------


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
    """Flatten a span's attribute list, applying the pre-v1.36 aliases."""
    attrs: Dict[str, Any] = {}
    for item in span.get('attributes', []):
        key = item.get('key')
        if not key:
            continue
        attrs[_LEGACY_ALIASES.get(key, key)] = _attr_value(item.get('value', {}))
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


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


def _usage(attrs: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Build ``token_usage`` from ``gen_ai.usage.*`` when any count exists."""
    prompt = attrs.get(_USAGE_INPUT)
    completion = attrs.get(_USAGE_OUTPUT)
    if prompt is None and completion is None:
        return None
    usage = {'input': int(prompt or 0), 'output': int(completion or 0)}
    usage['total'] = usage['input'] + usage['output']
    return usage


def _span_error_message(span: Dict[str, Any]) -> Optional[str]:
    """Return a message when the span records an error, else ``None``.

    Two sources: ``status.code == ERROR`` (with ``status.message`` when set)
    and span ``events`` whose ``name`` is ``exception``.
    """
    status = span.get('status') or {}
    try:
        code = int(status.get('code') or 0)
    except (TypeError, ValueError):
        code = 0
    if code == _STATUS_ERROR:
        return str(status.get('message') or 'span status is ERROR')
    for event in span.get('events', []):
        if event.get('name') == 'exception':
            attrs = {item.get('key'): _attr_value(item.get('value', {})) for item in event.get('attributes', [])}
            return str(attrs.get('exception.message') or attrs.get('exception.type') or 'span exception')
    return None


# ---------------------------------------------------------------------------
# ChatMessage reconstruction
# ---------------------------------------------------------------------------


def _part_to_message(role: str, part: Dict[str, Any], model: Optional[str]) -> Optional[ChatMessage]:
    """Convert one ``gen_ai.*.messages[].parts[]`` entry into a ChatMessage."""
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
    # Unknown part type (blob / uri / vendor extension): keep it visible in a
    # system message instead of dropping content silently.
    return ChatMessageSystem(content=_json_dumps({'type': part_type, 'raw': part}))


def _messages_from_attr(value: Any, model: Optional[str]) -> List[ChatMessage]:
    """Expand one ``gen_ai.*.messages`` attribute into ChatMessage objects.

    An assistant entry becomes **one** message carrying both its text and its
    tool calls, mirroring the bridge (``trace_recorder._build_assistant_message``).
    Emitting one message per part would turn ``[text, tool_call]`` into two
    consecutive assistant turns, and a prediction scan would mistake the
    trailing tool-call fragment for a missing final answer.
    """
    if not isinstance(value, list):
        return []
    messages: List[ChatMessage] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        role = entry.get('role', 'user')
        parts = entry.get('parts')
        if not isinstance(parts, list) or not parts:
            content = entry.get('content')
            if content is None:
                continue
            parts = [{'type': 'text', 'content': content}]

        if role == 'assistant':
            texts: List[str] = []
            tool_calls: List[ToolCall] = []
            for part in parts:
                part_type = part.get('type', 'text')
                if part_type == 'text':
                    texts.append(str(part.get('content', '')))
                elif part_type == 'tool_call':
                    tool_calls.append(_tool_call_from_part(part))
                # Unknown assistant parts (blob/uri/vendor) are not text and not
                # tool calls: skip rather than split the turn.
            messages.append(
                ChatMessageAssistant(
                    content='\n'.join(t for t in texts if t),
                    model=model,
                    tool_calls=tool_calls or None,
                )
            )
            continue

        for part in parts:
            part_type = part.get('type', 'text')
            if part_type == 'tool_call':
                # A tool_call part under a non-assistant role is malformed; keep
                # it visible as an assistant turn rather than dropping it.
                messages.append(ChatMessageAssistant(content='', model=model, tool_calls=[_tool_call_from_part(part)]))
                continue
            message = _part_to_message(role, part, model)
            if message is not None:
                messages.append(message)
    return messages


def _tool_call_from_part(part: Dict[str, Any]) -> ToolCall:
    """Build a ``ToolCall`` from one ``parts[]`` entry."""
    arguments = part.get('arguments')
    if not isinstance(arguments, (dict, str)):
        arguments = _json_dumps(arguments)
    return ToolCall(
        id=str(part.get('id') or ''),
        function=ToolFunction(name=str(part.get('name') or ''), arguments=arguments),
    )


def _tool_calls_in_messages_attr(value: Any) -> List[ToolCall]:
    """Extract ``tool_call`` parts from one ``gen_ai.*.messages`` attribute."""
    if not isinstance(value, list):
        return []
    calls: List[ToolCall] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        for part in entry.get('parts') or []:
            if isinstance(part, dict) and part.get('type') == 'tool_call':
                calls.append(_tool_call_from_part(part))
    return calls


def _legacy_input_messages(attrs: Dict[str, Any]) -> List[ChatMessage]:
    """Reconstruct the user prompt from the pre-v1.36 flat ``gen_ai.prompts``.

    Existing instrumentations keep emitting these by default, so accepting them
    avoids telling users to enable content capture when the real problem is a
    naming change. Only the prompt side is read here; the completion is added
    by :func:`_legacy_output_messages` so a legacy span does not append its
    assistant turn twice.
    """
    prompts = attrs.get(_LEGACY_PROMPTS)
    if isinstance(prompts, str):
        return [ChatMessageUser(content=prompts)]
    if isinstance(prompts, list):
        return [ChatMessageUser(content=str(item)) for item in prompts]
    return []


def _legacy_output_messages(attrs: Dict[str, Any], model: Optional[str]) -> List[ChatMessage]:
    """Reconstruct the assistant completion from the pre-v1.36 flat ``gen_ai.completions``."""
    completions = attrs.get(_LEGACY_COMPLETIONS)
    if isinstance(completions, str):
        return [ChatMessageAssistant(content=completions, model=model)]
    if isinstance(completions, list):
        return [ChatMessageAssistant(content=str(item), model=model) for item in completions]
    return []


def _message_signature(message: ChatMessage) -> tuple:
    """Stable identity for de-duplicating re-sent history across chat spans."""
    tool_ids = tuple(call.id for call in getattr(message, 'tool_calls', None) or [])
    return (type(message).__name__, getattr(message, 'text', None), tool_ids)


def _new_messages_tail(collected: List[ChatMessage], history: List[ChatMessage]) -> List[ChatMessage]:
    """Return the part of ``history`` that follows ``collected``.

    Each chat span repeats the whole conversation so far. Aligning the new
    history against what has already been collected recovers the tool
    observations that only appear in the next span's input.
    """
    if not history:
        return []
    if not collected:
        return history
    # Longest suffix of ``collected`` that is a prefix of ``history``.
    collected_sigs = [_message_signature(m) for m in collected]
    history_sigs = [_message_signature(m) for m in history]
    for overlap in range(min(len(collected_sigs), len(history_sigs)), 0, -1):
        if collected_sigs[-overlap:] == history_sigs[:overlap]:
            return history[overlap:]
    # No alignment (the exporter trims history): return everything new by
    # signature so nothing is silently dropped.
    seen = set(collected_sigs)
    return [message for message, sig in zip(history, history_sigs) if sig not in seen]


def _merge_assistant_tool_calls(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Fold consecutive assistant tool-call messages into one assistant turn."""
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


# ---------------------------------------------------------------------------
# Step resolution
# ---------------------------------------------------------------------------


def _owning_sample_id(span: Dict[str, Any], by_id: Dict[str, Dict[str, Any]], attribute: str) -> Optional[str]:
    """Resolve the sample id for ``span``, walking up to the nearest tagged ancestor.

    A multi-sample export usually tags only the trace root; resolving by
    ancestry keeps its chat / tool children in the same sample.
    """
    current: Optional[Dict[str, Any]] = span
    seen = set()
    while current is not None:
        span_id = _span_id(current)
        if span_id in seen:
            return None
        seen.add(span_id)
        value = _span_attributes(current).get(attribute)
        if value is not None:
            return str(value)
        current = by_id.get(current.get('parentSpanId') or '')
    return None


def _chat_steps(ordered: List[Dict[str, Any]]) -> Dict[str, int]:
    """Map span id → turn index.

    ``AgentTraceEvent.step`` is documented as the "0-based loop iteration", not
    the span nesting depth: the i-th ``chat`` span (in start-time order) is
    turn ``i``. Tool spans inherit the turn of their nearest ``chat`` ancestor;
    agent-level spans use 0.
    """
    by_id = {_span_id(span): span for span in ordered}
    step_of: Dict[str, int] = {}
    turn = 0
    for span in ordered:
        if _span_attributes(span).get(_OP_NAME) in (_OP_CHAT, _OP_TEXT_COMPLETION):
            step_of[_span_id(span)] = turn
            turn += 1

    def resolve(span: Dict[str, Any]) -> int:
        span_id = _span_id(span)
        if span_id in step_of:
            return step_of[span_id]
        seen = {span_id}
        current = span
        while True:
            parent = by_id.get(current.get('parentSpanId') or '')
            if parent is None or _span_id(parent) in seen:
                return 0
            parent_id = _span_id(parent)
            if parent_id in step_of:
                return step_of[parent_id]
            seen.add(parent_id)
            current = parent

    return {_span_id(span): resolve(span) for span in ordered}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
        expected_sample_id: When set, the export must contain at least one span
            whose ``sample_id_attribute`` equals this value; spans for other
            samples are dropped.

    Returns:
        The ``AgentTrace`` (``framework='otel-import'``) and the reconstructed
        message history, in the shape ``InferenceResult`` expects.

    Raises:
        OTLPImportError: The export is empty, has no GenAI chat spans, is
            missing message content, or cannot be associated with a sample.
    """
    spans = _iter_spans(payload)
    if not spans:
        raise OTLPImportError('OTLP export contains no spans')

    # --- Sample association -------------------------------------------------
    if expected_sample_id is not None:
        # Sample ids are integers in the dataset but may be int or str in OTLP
        # attributes. Compare string forms, and resolve the owning id by
        # walking to the trace root: a multi-sample export normally tags only
        # the invoke_agent root, so a per-span check would drop its children.
        expected = str(expected_sample_id)
        by_id = {_span_id(span): span for span in spans}
        resolved = {_span_id(span): _owning_sample_id(span, by_id, sample_id_attribute) for span in spans}
        tagged = {value for value in resolved.values() if value is not None}
        if not tagged:
            raise OTLPImportError(
                f'No span carries {sample_id_attribute!r}; cannot associate the export with '
                f'sample {expected_sample_id!r}'
            )
        if expected not in tagged:
            raise OTLPImportError(
                f'No span carries {sample_id_attribute}={expected_sample_id!r}; found sample ids: {sorted(tagged)}'
            )
        spans = [span for span in spans if resolved.get(_span_id(span)) == expected]

    ordered = sorted(spans, key=_start_ns)
    steps = _chat_steps(ordered)

    trace = AgentTrace(framework=FRAMEWORK)
    # ``create_agent`` is setup, not an invocation, so it emits no event; its
    # identifying attributes ride along on the invoke_agent RUN_START payload.
    agent_identity: Dict[str, Any] = {
        key: value
        for span in ordered
        if _span_attributes(span).get(_OP_NAME) == _OP_CREATE_AGENT
        for key, value in _span_attributes(span).items()
        if key in _AGENT_IDENTITY_KEYS
    }
    messages: List[ChatMessage] = []
    skipped_operations: Counter = Counter()
    chat_turns = 0
    saw_content = False
    tool_calls_from_messages: set = set()
    emitted_tool_results: set = set()

    # First pass: reconstruct messages, so TOOL_CALL dedup knows which ids the
    # message history already carries.
    for span in ordered:
        attrs = _span_attributes(span)
        if attrs.get(_OP_NAME) not in (_OP_CHAT, _OP_TEXT_COMPLETION):
            continue
        if chat_turns == 0:
            input_messages = _messages_from_attr(
                attrs.get(_INPUT_MESSAGES), attrs.get('gen_ai.request.model')
            ) or _legacy_input_messages(attrs)
            if input_messages:
                messages.extend(input_messages)
                saw_content = True
        else:
            # ``gen_ai.input.messages`` is the full history: later chat spans
            # re-send everything already collected plus the new tool results.
            # Append only the tail so tool observations are not lost.
            history = _messages_from_attr(
                attrs.get(_INPUT_MESSAGES), attrs.get('gen_ai.request.model')
            ) or _legacy_input_messages(attrs)
            new_tail = _new_messages_tail(messages, history)
            if new_tail:
                messages.extend(new_tail)
                saw_content = True
        output_messages = _messages_from_attr(
            attrs.get(_OUTPUT_MESSAGES), attrs.get('gen_ai.response.model')
        ) or _legacy_output_messages(attrs, attrs.get('gen_ai.response.model'))
        if output_messages:
            messages.extend(output_messages)
            saw_content = True
        chat_turns += 1

    for message in messages:
        if isinstance(message, ChatMessageAssistant) and message.tool_calls:
            for call in message.tool_calls:
                if call.id:
                    tool_calls_from_messages.add(call.id)

    # Second pass: events.
    for span in ordered:
        span_id = _span_id(span)
        attrs = _span_attributes(span)
        operation = attrs.get(_OP_NAME)
        if operation is None:
            raise OTLPImportError(f'Span {span_id!r} has no {_OP_NAME!r}; only GenAI operation spans can be imported')
        step = steps.get(span_id, 0)
        payload_extra = {k: v for k, v in attrs.items() if k not in _KNOWN_ATTRIBUTES}

        error_message = _span_error_message(span)
        if error_message is not None:
            trace.add_event(
                step=step,
                type=EventType.ERROR,
                latency_ms=_latency_ms(span),
                payload={'span_id': span_id, 'operation': operation, 'message': error_message, **payload_extra},
            )

        if operation in (_OP_CHAT, _OP_TEXT_COMPLETION):
            trace.add_event(
                step=step,
                type=EventType.MODEL_GENERATE,
                latency_ms=_latency_ms(span),
                token_usage=_usage(attrs),
                payload={'span_id': span_id, 'operation': operation, **payload_extra},
            )
            # A tool call encoded as a ``tool_call`` part belongs to this turn;
            # the bridge and native loop record one TOOL_CALL per assistant
            # tool call, so emit it here rather than relying on the tool span.
            for call in _tool_calls_in_messages_attr(attrs.get(_OUTPUT_MESSAGES)):
                trace.add_event(
                    step=step,
                    type=EventType.TOOL_CALL,
                    payload={
                        'span_id': span_id,
                        'operation': operation,
                        'tool_call_id': call.id,
                        'name': call.function.name,
                        'arguments': call.function.arguments,
                    },
                )
            continue

        if operation == _OP_EXECUTE_TOOL:
            tool_name = attrs.get('gen_ai.tool.name') or span.get('name')
            tool_args = attrs.get('gen_ai.tool.call.arguments')
            tool_result = attrs.get('gen_ai.tool.call.result')
            call_id = str(attrs.get('gen_ai.tool.call.id') or span_id)
            # Messages are the source of truth for TOOL_CALL; synthesise one only
            # when the message history has no matching part.
            if call_id not in tool_calls_from_messages:
                trace.add_event(
                    step=step,
                    type=EventType.TOOL_CALL,
                    payload={
                        'span_id': span_id,
                        'operation': operation,
                        'tool_call_id': call_id,
                        'name': tool_name,
                        'arguments': tool_args,
                        **payload_extra,
                    },
                )
            if call_id not in emitted_tool_results:
                trace.add_event(
                    step=step,
                    type=EventType.TOOL_RESULT,
                    latency_ms=_latency_ms(span),
                    payload={
                        'span_id': span_id,
                        'operation': operation,
                        'tool_call_id': call_id,
                        'result': tool_result,
                        **payload_extra,
                    },
                )
                emitted_tool_results.add(call_id)
            continue

        if operation == _OP_INVOKE_AGENT:
            # The span's own identity wins over the create_agent association.
            identity = {**agent_identity, **{k: v for k, v in payload_extra.items() if k in _AGENT_IDENTITY_KEYS}}
            trace.add_event(
                step=step,
                type=EventType.RUN_START,
                payload={'span_id': span_id, 'operation': operation, **payload_extra, **identity},
            )
            trace.add_event(
                step=step,
                type=EventType.RUN_END,
                latency_ms=_latency_ms(span),
                payload={'span_id': span_id, 'operation': operation},
            )
            continue

        if operation == _OP_CREATE_AGENT:
            # Constructing an agent is setup, not an invocation: no run boundary.
            # Its identifying attributes ride along on the invoke_agent payload.
            continue

        # Valid-but-uninteresting (embeddings / generate_content) or vendor op:
        # skip and count, so a healthy RAG trace is not marked as an error.
        skipped_operations[operation] += 1

    if chat_turns == 0:
        raise OTLPImportError(
            f'No {_OP_CHAT!r} spans found; cannot reconstruct a message history '
            f'(operations seen: {sorted({_span_attributes(s).get(_OP_NAME) for s in spans})})'
        )
    if not saw_content:
        raise OTLPImportError(
            f'No message content found in {chat_turns} chat span(s). GenAI semantic conventions make '
            f'content capture opt-in; enable it on the exporter so {_INPUT_MESSAGES!r} / '
            f'{_OUTPUT_MESSAGES!r} (or the legacy {_LEGACY_PROMPTS!r} / {_LEGACY_COMPLETIONS!r}) '
            f'are emitted'
        )

    if skipped_operations:
        # Attach to the existing RUN_END payload: adding another RUN_END event
        # would grow a second run boundary on a trace that already has one.
        for event in reversed(trace.events):
            if event.type is EventType.RUN_END:
                event.payload = {**event.payload, 'skipped_operations': dict(skipped_operations)}
                break

    return trace, _merge_assistant_tool_calls(messages)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def load_otlp_trace_for_sample(config: Any, model: Any, sample: Any) -> Any:
    """Load the OTLP export for one sample and turn it into an ``InferenceResult``.

    ``config.trace_path`` may be one OTLP JSON file (one export per sample) or
    a directory containing ``<sample_id>.otlp.json``. When the export carries
    ``config.sample_id_attribute`` on any span, it is treated as multi-sample
    and filtered by ``sample.id``; otherwise the file is taken as this
    sample's export.

    Raises:
        OTLPImportError: The export is missing, has no content, cannot be
            associated with the sample, or ends on a tool call without a final
            answer.
    """
    import json
    from pathlib import Path

    from evalscope.api.evaluator import InferenceResult
    from evalscope.api.messages import ChatMessageAssistant
    from evalscope.api.model import ModelOutput

    path = Path(config.trace_path)
    if path.is_dir():
        candidate = path / f'{sample.id}.otlp.json'
        if not candidate.exists():
            raise OTLPImportError(f'No OTLP export for sample {sample.id!r} in {path} (expected {candidate.name})')
        path = candidate

    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise OTLPImportError(f'OTLP export not found: {path}') from exc
    except json.JSONDecodeError as exc:
        raise OTLPImportError(f'OTLP export at {path} is not valid JSON: {exc}') from exc
    expected = None
    if sample.id is not None and any(
        _span_attributes(span).get(config.sample_id_attribute) is not None for span in _iter_spans(payload)
    ):
        expected = str(sample.id)
    trace, messages = import_otlp_trace(
        payload,
        sample_id_attribute=config.sample_id_attribute,
        expected_sample_id=expected,
    )

    # The prediction artifact is the last assistant turn that carries text. A
    # trace that ends on a tool call has no final answer: fail loudly instead of
    # scoring an intermediate sentence.
    prediction: Optional[str] = None
    for message in reversed(messages):
        if isinstance(message, ChatMessageAssistant):
            text = getattr(message, 'text', None)
            if text:
                prediction = text
                break
            if message.tool_calls:
                raise OTLPImportError(
                    f'Trace for sample {sample.id!r} ends on a tool call without a final answer; '
                    f'cannot score an empty prediction'
                )
    if prediction is None:
        raise OTLPImportError(f'Trace for sample {sample.id!r} has no assistant text to use as the prediction')

    output = ModelOutput.from_content(model=f'{model.name}-otel-import', content=prediction)
    return InferenceResult(output=output, messages=messages, trace=trace)
