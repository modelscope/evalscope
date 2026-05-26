"""Anthropic Messages API ⇄ EvalScope native type translation.

P0 scope: text + ``tool_use`` + ``tool_result`` blocks, no streaming, no
``cache_control``, no extended-thinking blocks.  See
``.qoder/plans/agent_bridge_design.md`` §7.4 for known-lossy cases.
"""

import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evalscope.api.messages import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction, ToolInfo, ToolParams


def unpack_tool_call(tool_call: Any) -> Tuple[str, Dict[str, Any]]:
    """Return ``(name, arguments)`` for a :class:`ToolCall`.

    ``ToolCall.function`` is normally a :class:`ToolFunction` but some
    upstream paths leave it as a bare string; handle both shapes here so
    every emitter / translator can call this instead of re-doing the
    isinstance dance.
    """
    fn = tool_call.function
    if isinstance(fn, ToolFunction):
        return fn.name, fn.arguments or {}
    return str(fn), {}


def anthropic_request_to_messages(body: Dict[str, Any]) -> List[ChatMessage]:
    """Convert an Anthropic Messages request body into a flat ChatMessage list.

    Anthropic stores the system prompt out-of-band on the top-level ``system``
    field; we prepend it as a ``ChatMessageSystem`` so the EvalScope model
    layer sees a single ordered transcript.  ``tool_result`` blocks inside
    user messages become standalone ``ChatMessageTool`` entries to match
    OpenAI's transcript shape.
    """
    messages: List[ChatMessage] = []
    system = body.get('system')
    if isinstance(system, str) and system:
        messages.append(ChatMessageSystem(content=system))
    elif isinstance(system, list):
        text = ''.join(b.get('text', '') for b in system if isinstance(b, dict) and b.get('type') == 'text')
        if text:
            messages.append(ChatMessageSystem(content=text))

    for entry in body.get('messages') or []:
        if not isinstance(entry, dict):
            continue
        role = entry.get('role')
        content = entry.get('content')
        if role == 'user':
            messages.extend(_user_blocks_to_messages(content))
        elif role == 'assistant':
            messages.append(_assistant_blocks_to_message(content))
    return messages


def _user_blocks_to_messages(content: Any) -> List[ChatMessage]:
    if isinstance(content, str):
        return [ChatMessageUser(content=content)]
    if not isinstance(content, list):
        return []
    user_text_parts: List[str] = []
    tool_msgs: List[ChatMessage] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get('type')
        if btype == 'text':
            user_text_parts.append(block.get('text', ''))
        elif btype == 'tool_result':
            tool_msgs.append(_tool_result_to_message(block))
    # Tool results precede any new user text so the model sees the
    # observation first, then the new prompt (matches OpenAI ordering).
    out: List[ChatMessage] = list(tool_msgs)
    if user_text_parts:
        out.append(ChatMessageUser(content='\n'.join(p for p in user_text_parts if p)))
    return out


def _tool_result_to_message(block: Dict[str, Any]) -> ChatMessageTool:
    raw = block.get('content', '')
    if isinstance(raw, list):
        text = ''.join(b.get('text', '') for b in raw if isinstance(b, dict) and b.get('type') == 'text')
    else:
        text = str(raw)
    is_error = bool(block.get('is_error', False))
    error = ToolCallError(type='unknown', message=text) if is_error else None
    return ChatMessageTool(
        content=text,
        tool_call_id=block.get('tool_use_id'),
        error=error,
    )


def _assistant_blocks_to_message(content: Any) -> ChatMessageAssistant:
    if isinstance(content, str):
        return ChatMessageAssistant(content=content)
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get('type')
            if btype == 'text':
                text_parts.append(block.get('text', ''))
            elif btype == 'tool_use':
                tool_calls.append(
                    ToolCall(
                        id=block.get('id') or f'toolu_{uuid.uuid4().hex[:12]}',
                        function=ToolFunction(
                            name=block.get('name', ''),
                            arguments=block.get('input') or {},
                        ),
                        type='function',
                    )
                )
    return ChatMessageAssistant(
        content='\n'.join(p for p in text_parts if p),
        tool_calls=tool_calls or None,
    )


def anthropic_tools_to_tool_infos(tools: Sequence[Dict[str, Any]]) -> List[ToolInfo]:
    """Translate Anthropic tool specs to ``ToolInfo``.  Best-effort: any
    unparsable parameter schema falls back to an empty ``ToolParams``."""
    out: List[ToolInfo] = []
    for spec in tools or []:
        if not isinstance(spec, dict):
            continue
        name = spec.get('name')
        if not name:
            continue
        schema = spec.get('input_schema') or {}
        params = ToolParams() if not isinstance(schema, dict) else _safe_tool_params(schema)
        out.append(ToolInfo(
            name=name,
            description=spec.get('description', '') or '',
            parameters=params,
        ))
    return out


def _safe_tool_params(schema: Dict[str, Any]) -> ToolParams:
    try:
        # ToolParams.type is Literal['object']; only forward properties/required.
        return ToolParams.model_validate({
            'properties': schema.get('properties', {}) or {},
            'required': schema.get('required', []) or [],
        })
    except Exception:
        return ToolParams()


def model_output_to_anthropic_response(
    output: ModelOutput,
    *,
    request_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a :class:`ModelOutput` as an Anthropic Messages response."""
    message = output.message if output.choices else None
    blocks: List[Dict[str, Any]] = []
    if message is not None:
        text = message.text or ''
        if text:
            blocks.append({'type': 'text', 'text': text})
        for tc in message.tool_calls or []:
            name, args = unpack_tool_call(tc)
            blocks.append({
                'type': 'tool_use',
                'id': tc.id,
                'name': name,
                'input': args,
            })
    if not blocks:
        blocks.append({'type': 'text', 'text': ''})

    stop_reason = map_stop_reason_to_anthropic(output.choices[0].stop_reason if output.choices else 'stop')
    usage = output.usage
    return {
        'id': output.id or f'msg_{uuid.uuid4().hex[:24]}',
        'type': 'message',
        'role': 'assistant',
        'content': blocks,
        'model': request_model or output.model or '',
        'stop_reason': stop_reason,
        'stop_sequence': None,
        'usage': {
            'input_tokens': usage.input_tokens if usage else 0,
            'output_tokens': usage.output_tokens if usage else 0,
        },
    }


_STOP_REASON_MAP = {
    'stop': 'end_turn',
    'max_tokens': 'max_tokens',
    'tool_calls': 'tool_use',
    'model_length': 'max_tokens',
    'content_filter': 'end_turn',
    'unknown': 'end_turn',
}


def map_stop_reason_to_anthropic(reason: str) -> str:
    """Translate an EvalScope ``StopReason`` to its Anthropic equivalent.

    Public helper (used by :mod:`.sse_anthropic` too) so the mapping
    table lives in exactly one place.
    """
    return _STOP_REASON_MAP.get(reason, 'end_turn')
