"""OpenAI Chat Completions API ⇄ EvalScope native type translation.

P0 scope: text + ``tool_calls`` + ``tool`` role messages + vendor-extension
``reasoning_content`` (DashScope / DeepSeek). Multi-modal content blocks
(image / audio) are passed through best-effort as plain text. No JSON
``response_format`` enforcement (transparent passthrough only).

The bridge is a protocol translator — it converts the agent's wire format
into ``ChatMessage`` so :meth:`Model.generate_async` (which may dispatch
to any backend, including Anthropic) handles the request, then renders
the resulting :class:`ModelOutput` back as OpenAI shape.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evalscope.api.messages import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolCall, ToolChoice, ToolFunction, ToolInfo, ToolParams


def openai_request_to_messages(body: Dict[str, Any]) -> List[ChatMessage]:
    """Convert an OpenAI Chat Completions request body to ``ChatMessage[]``.

    Preserves message order. Tool-call requests on assistant messages and
    their matching ``role:'tool'`` results are kept as separate transcript
    entries (matches EvalScope's native shape).
    """
    messages: List[ChatMessage] = []
    for entry in body.get('messages') or []:
        if not isinstance(entry, dict):
            continue
        role = entry.get('role')
        if role == 'system':
            messages.append(ChatMessageSystem(content=_extract_text(entry.get('content'))))
        elif role == 'user':
            messages.append(ChatMessageUser(content=_extract_text(entry.get('content'))))
        elif role == 'assistant':
            messages.append(_assistant_entry_to_message(entry))
        elif role == 'tool':
            messages.append(_tool_entry_to_message(entry))
    return messages


def _extract_text(content: Any) -> str:
    """Flatten an OpenAI message content (str or content-parts list) to text.

    Non-text parts (image_url, audio, etc.) are dropped — we do not yet
    surface them in the bridge translation path.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get('type')
            if btype == 'text' or btype == 'input_text':
                parts.append(block.get('text', '') or '')
        return '\n'.join(p for p in parts if p)
    return ''


def _assistant_entry_to_message(entry: Dict[str, Any]) -> ChatMessageAssistant:
    text = _extract_text(entry.get('content'))
    reasoning = entry.get('reasoning_content') or entry.get('reasoning')
    tool_calls_in = entry.get('tool_calls') or []
    tool_calls: List[ToolCall] = []
    for raw in tool_calls_in:
        if not isinstance(raw, dict):
            continue
        fn = raw.get('function') or {}
        name = fn.get('name', '') or ''
        args = _parse_tool_arguments(fn.get('arguments'))
        tool_calls.append(
            ToolCall(
                id=raw.get('id') or f'call_{uuid.uuid4().hex[:24]}',
                function=ToolFunction(name=name, arguments=args),
                type='function',
            )
        )

    if reasoning:
        # reasoning lives as a typed ContentReasoning block in front of the text.
        content_blocks: List[Any] = [ContentReasoning(reasoning=str(reasoning))]
        if text:
            content_blocks.append(ContentText(text=text))
        return ChatMessageAssistant(content=content_blocks, tool_calls=tool_calls or None)
    return ChatMessageAssistant(content=text, tool_calls=tool_calls or None)


def _tool_entry_to_message(entry: Dict[str, Any]) -> ChatMessageTool:
    text = _extract_text(entry.get('content'))
    return ChatMessageTool(
        content=text,
        tool_call_id=entry.get('tool_call_id'),
        function=entry.get('name'),
    )


def _parse_tool_arguments(raw: Any) -> Dict[str, Any]:
    """Decode ``function.arguments`` (a JSON string per OpenAI spec) to dict.

    Vendor responses occasionally wrap arguments in markdown fences or
    include trailing commas; fall back to ``{"_raw": raw}`` so the
    downstream :class:`ToolFunction` validator does not blow up the
    whole turn over a malformed payload.
    """
    if raw is None or raw == '':
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {'_raw': raw}
        if isinstance(parsed, dict):
            return parsed
        return {'_raw': raw}
    return {}


def openai_tools_to_tool_infos(tools: Sequence[Dict[str, Any]]) -> List[ToolInfo]:
    """Translate OpenAI ``tools[]`` (with the ``function`` wrapper) to ``ToolInfo``."""
    out: List[ToolInfo] = []
    for spec in tools or []:
        if not isinstance(spec, dict):
            continue
        if spec.get('type') and spec.get('type') != 'function':
            # only "function" is supported on this path
            continue
        fn = spec.get('function') or {}
        name = fn.get('name')
        if not name:
            continue
        schema = fn.get('parameters') or {}
        params = _safe_tool_params(schema) if isinstance(schema, dict) else ToolParams()
        out.append(ToolInfo(
            name=name,
            description=fn.get('description', '') or '',
            parameters=params,
        ))
    return out


def _safe_tool_params(schema: Dict[str, Any]) -> ToolParams:
    try:
        return ToolParams.model_validate({
            'properties': schema.get('properties', {}) or {},
            'required': schema.get('required', []) or [],
        })
    except Exception:
        return ToolParams()


def openai_tool_choice(raw: Any) -> Optional[ToolChoice]:
    """Map OpenAI ``tool_choice`` to EvalScope :class:`ToolChoice`.

    ``None`` (default) leaves the choice up to the model layer. ``'required'``
    maps to ``'any'`` per EvalScope semantics (model must call at least one).
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw == 'auto':
            return 'auto'
        if raw == 'none':
            return 'none'
        if raw == 'required':
            return 'any'
        return None
    if isinstance(raw, dict):
        if raw.get('type') != 'function':
            return None
        fn = raw.get('function') or {}
        name = fn.get('name')
        if name:
            return ToolFunction(name=name, arguments={})
    return None


def model_output_to_openai_response(
    output: ModelOutput,
    *,
    request_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a :class:`ModelOutput` as an OpenAI ``chat.completion`` response."""
    message = output.message if output.choices else None
    msg_payload: Dict[str, Any] = {'role': 'assistant', 'content': None}
    if message is not None:
        text, reasoning = _split_text_and_reasoning(message)
        msg_payload['content'] = text if text else None
        if reasoning:
            msg_payload['reasoning_content'] = reasoning
        if message.tool_calls:
            msg_payload['tool_calls'] = [_render_tool_call(tc) for tc in message.tool_calls]

    stop_reason = output.choices[0].stop_reason if output.choices else 'stop'
    finish_reason = map_stop_reason_to_openai(stop_reason)
    usage = output.usage
    return {
        'id': output.id or f'chatcmpl-{uuid.uuid4().hex[:24]}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': request_model or output.model or '',
        'choices': [{
            'index': 0,
            'message': msg_payload,
            'finish_reason': finish_reason,
        }],
        'usage': {
            'prompt_tokens': usage.input_tokens if usage else 0,
            'completion_tokens': usage.output_tokens if usage else 0,
            'total_tokens': usage.total_tokens if usage else 0,
        },
    }


def _render_tool_call(tc: ToolCall) -> Dict[str, Any]:
    name, args = unpack_openai_tool_call(tc)
    return {
        'id': tc.id,
        'type': 'function',
        'function': {
            'name': name,
            'arguments': json.dumps(args, ensure_ascii=False),
        },
    }


def unpack_openai_tool_call(tool_call: ToolCall) -> Tuple[str, Dict[str, Any]]:
    """Return ``(name, arguments)`` for a :class:`ToolCall`. Handles the
    edge case where ``function`` is a bare string instead of :class:`ToolFunction`."""
    fn = tool_call.function
    if isinstance(fn, ToolFunction):
        return fn.name, fn.arguments or {}
    return str(fn), {}


def _split_text_and_reasoning(message: ChatMessageAssistant) -> Tuple[str, str]:
    """Pull text + reasoning out of the assistant message's content blocks.

    ``ChatMessageAssistant.content`` is either a plain string or a list of
    typed content blocks. Reasoning blocks are concatenated and returned
    separately so we can emit ``reasoning_content`` alongside ``content``.
    """
    if isinstance(message.content, str):
        return message.content, ''
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    for block in message.content:
        btype = getattr(block, 'type', None)
        if btype == 'text':
            text_parts.append(getattr(block, 'text', '') or '')
        elif btype == 'reasoning':
            reasoning_parts.append(getattr(block, 'reasoning', '') or '')
    return '\n'.join(p for p in text_parts if p), '\n'.join(p for p in reasoning_parts if p)


_STOP_REASON_MAP = {
    'stop': 'stop',
    'tool_calls': 'tool_calls',
    'max_tokens': 'length',
    'model_length': 'length',
    'content_filter': 'content_filter',
    'unknown': 'stop',
}


def map_stop_reason_to_openai(reason: str) -> str:
    """Translate an EvalScope ``StopReason`` to its OpenAI ``finish_reason``."""
    return _STOP_REASON_MAP.get(reason, 'stop')
