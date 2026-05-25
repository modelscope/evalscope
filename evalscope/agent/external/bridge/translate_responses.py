"""OpenAI Responses API ⇄ EvalScope native type translation.

The Responses API (``/v1/responses``) is OpenAI's newer protocol used by
codex CLI v0.133+ — chat completions support was removed and Responses
is the only wire format codex will speak. Unlike chat completions, the
request carries a typed ``input[]`` array (items of type ``message`` /
``function_call`` / ``function_call_output`` / ``reasoning``) instead
of a flat ``messages[]`` list, and ``instructions`` lives at the top
level rather than a system message.

P0 scope: text + ``function_call`` + ``function_call_output`` +
``reasoning`` (summary). Vision / web_search / computer_call / MCP
items are not translated (codex doesn't emit them; raise WARN if seen
in the wild).

This module is a sibling of :mod:`translate_anthropic` /
:mod:`translate_openai`. Several helpers from the OpenAI chat module
are reused verbatim (tool argument parsing, reasoning splitting,
tool_choice mapping) since both protocols share OpenAI semantics for
those concepts.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

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
from evalscope.api.tool import ToolCall, ToolChoice, ToolFunction, ToolInfo
from evalscope.utils.logger import get_logger
from .translate_openai import (
    _parse_tool_arguments,
    _safe_tool_params,
    _split_text_and_reasoning,
    openai_tool_choice,
    unpack_openai_tool_call,
)

logger = get_logger()

# ---------------------------------------------------------------------------
# Request: Responses input[] → ChatMessage[]
# ---------------------------------------------------------------------------

# Item types treated as "user-triggered tool outputs" — they share the
# function_call_output handling path (mapped to ChatMessageTool).
_TOOL_OUTPUT_ITEM_TYPES = frozenset({
    'function_call_output',
    'custom_tool_call_output',
    'computer_call_output',
    'local_shell_call_output',
})

# Item types treated as "assistant-side function calls" — they share the
# function_call handling path (mapped to ToolCall on the pending assistant).
_TOOL_CALL_ITEM_TYPES = frozenset({
    'function_call',
    'custom_tool_call',
})

# Item types that represent built-in tool use the agent CLI initiated
# itself (web search, MCP, file search, etc.). The downstream model
# (typically a chat-only LLM like qwen3-max) can't actually invoke
# these, but it should *see* that they happened so multi-turn context
# stays coherent. Rendered as opaque placeholder text on the pending
# assistant message — never a ChatMessageTool, since there's no
# matching tool_call_id on our side.
_BUILTIN_ASSISTANT_ITEM_TYPES = frozenset({
    'computer_call',
    'web_search_call',
    'file_search_call',
    'image_generation_call',
    'code_interpreter_call',
    'mcp_call',
    'mcp_list_tools',
    'mcp_approval_request',
    'mcp_approval_response',
    'local_shell_call',
})


def responses_request_to_messages(body: Dict[str, Any]) -> List[ChatMessage]:
    """Walk a Responses API request body into a flat ``ChatMessage`` list.

    Order: top-level ``instructions`` becomes a system message, then
    ``input[]`` items are translated in order. ``function_call`` /
    ``custom_tool_call`` items are merged into the immediately preceding
    assistant message (or start a new bare-tool-call assistant if none);
    ``reasoning`` items prepend a :class:`ContentReasoning` block to the
    next assistant message via a pending buffer.

    Built-in tool use items the agent CLI initiated itself
    (``web_search_call``, ``mcp_call``, ``file_search_call``,
    ``computer_call``, ``code_interpreter_call``, etc.) are rendered as
    opaque placeholder text on the pending assistant message so the
    downstream chat-only model still sees prior-turn context, even
    though it cannot directly invoke them.

    Unknown item types are WARNed and skipped (PR2 forward-compat: new
    codex / OpenAI item types won't crash the bridge, but the gap will
    surface in logs).
    """
    messages: List[ChatMessage] = []

    if instr := body.get('instructions'):
        messages.append(ChatMessageSystem(content=str(instr)))

    pending_assistant: Optional[ChatMessageAssistant] = None

    def _flush() -> None:
        nonlocal pending_assistant
        if pending_assistant is not None:
            messages.append(pending_assistant)
            pending_assistant = None

    def _append_assistant_block(block: Any) -> None:
        """Add a ContentText/ContentReasoning block to the pending assistant
        message, creating one if missing."""
        nonlocal pending_assistant
        if pending_assistant is None:
            pending_assistant = ChatMessageAssistant(content=[block])
            return
        existing = pending_assistant.content
        if isinstance(existing, list):
            pending_assistant.content = [*existing, block]
        elif existing:
            pending_assistant.content = [ContentText(text=existing), block]
        else:
            pending_assistant.content = [block]

    for item in body.get('input') or []:
        if not isinstance(item, dict):
            continue
        itype = item.get('type')

        if itype == 'message':
            role = item.get('role')
            text = _flatten_responses_content(item.get('content'))
            if role == 'system' or role == 'developer':
                # Responses spec uses 'developer' for what was 'system';
                # treat both as ChatMessageSystem.
                _flush()
                messages.append(ChatMessageSystem(content=text))
            elif role == 'user':
                _flush()
                messages.append(ChatMessageUser(content=text))
            elif role == 'assistant':
                # If the pending assistant only contains reasoning so far
                # (a CoT block that immediately precedes the actual answer),
                # merge the text into the same assistant message rather than
                # emitting two separate turns.
                if _pending_is_reasoning_only(pending_assistant):
                    if text:
                        _append_assistant_block(ContentText(text=text))
                else:
                    _flush()
                    pending_assistant = ChatMessageAssistant(content=text)

        elif itype in _TOOL_CALL_ITEM_TYPES:
            tc = _tool_call_from_responses_item(item)
            if pending_assistant is None:
                pending_assistant = ChatMessageAssistant(content='', tool_calls=[tc])
            else:
                pending_assistant.tool_calls = (pending_assistant.tool_calls or []) + [tc]

        elif itype in _TOOL_OUTPUT_ITEM_TYPES:
            _flush()
            messages.append(
                ChatMessageTool(
                    content=_flatten_function_call_output(item.get('output')),
                    tool_call_id=item.get('call_id'),
                )
            )

        elif itype == 'reasoning':
            text = _extract_reasoning_text(item)
            if not text:
                continue
            reasoning_block = ContentReasoning(reasoning=text)
            # reasoning typically *prepends* the upcoming assistant content
            # (chain-of-thought leading to the answer), so handle the
            # prepend case manually rather than reusing _append_assistant_block.
            if pending_assistant is None:
                pending_assistant = ChatMessageAssistant(content=[reasoning_block])
            else:
                existing = pending_assistant.content
                if isinstance(existing, list):
                    pending_assistant.content = [reasoning_block, *existing]
                elif existing:
                    pending_assistant.content = [reasoning_block, ContentText(text=existing)]
                else:
                    pending_assistant.content = [reasoning_block]

        elif itype in _BUILTIN_ASSISTANT_ITEM_TYPES:
            # Render as opaque placeholder so the downstream model sees
            # that something happened in the prior turn. We can't surface
            # the result as a ChatMessageTool because there's no matching
            # tool_call_id our side issued.
            placeholder = _summarize_builtin_tool_item(item)
            _append_assistant_block(ContentText(text=placeholder))

        elif itype == 'item_reference':
            # Pointer to a stored response item from a previous turn;
            # bridge is stateless, can't resolve. Log + drop.
            logger.warning(
                f'responses translator: dropping item_reference '
                f'(stateful sessions not supported); item={item!r}'
            )

        else:
            logger.warning(
                f'responses translator: ignoring unsupported input item type={itype!r} '
                f'(if this is a codex / OpenAI item we need to handle, file a bug)'
            )

    _flush()
    return messages


def _pending_is_reasoning_only(pending: Optional[ChatMessageAssistant]) -> bool:
    """True iff ``pending`` holds *only* one or more reasoning blocks
    (no text, no tool_calls). Used to decide whether a following
    assistant message item belongs to the same turn (CoT → answer)
    or should start a new turn."""
    if pending is None or pending.tool_calls:
        return False
    content = pending.content
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(b, ContentReasoning) for b in content)


def _tool_call_from_responses_item(item: Dict[str, Any]) -> ToolCall:
    """Build a :class:`ToolCall` from a ``function_call`` or
    ``custom_tool_call`` item. The two differ in argument shape:

    * ``function_call.arguments`` is a JSON-encoded string (per OpenAI spec)
    * ``custom_tool_call.input`` is a free-form string the model emitted
      verbatim; wrap as ``{'input': <string>}`` for tool-layer consumption
    """
    itype = item.get('type')
    name = item.get('name', '') or ''
    call_id = item.get('call_id') or f'call_{uuid.uuid4().hex[:24]}'
    if itype == 'custom_tool_call':
        arguments = {'input': item.get('input', '') or ''}
        tc_type = 'custom'
    else:
        arguments = _parse_tool_arguments(item.get('arguments'))
        tc_type = 'function'
    return ToolCall(
        id=call_id,
        function=ToolFunction(name=name, arguments=arguments),
        type=tc_type,
    )


def _summarize_builtin_tool_item(item: Dict[str, Any]) -> str:
    """Render a built-in tool item (web_search / mcp / file_search /
    computer / etc.) as a short opaque placeholder string.

    Goal: give the downstream model enough context to know prior turns
    used a built-in tool, without faithfully replaying every field.
    Format: ``[<type> id=<id> <summary>]``.
    """
    itype = item.get('type', 'unknown')
    item_id = item.get('id') or item.get('call_id') or '<no-id>'
    detail_parts: List[str] = []
    # Common fields worth showing if present.
    for key in ('name', 'server_label', 'query', 'status'):
        if (val := item.get(key)) is not None:
            detail_parts.append(f'{key}={val!r}')
    # Action / output payloads are often nested dicts; render type only.
    if isinstance(item.get('action'), dict):
        if action_type := item['action'].get('type'):
            detail_parts.append(f'action={action_type!r}')
    if (output := item.get('output')) is not None:
        out_str = str(output)
        detail_parts.append(f'output={out_str[:80]!r}' + ('…' if len(out_str) > 80 else ''))
    detail = ' '.join(detail_parts)
    return f'[{itype} id={item_id} {detail}]'.rstrip()


def _flatten_responses_content(content: Any) -> str:
    """Flatten a Responses ``message.content`` (list of content-parts) to text.

    Recognised part types: ``input_text`` (user input), ``output_text``
    (assistant reply replayed back), plain ``text``. Image / audio parts
    are dropped — bridge doesn't yet surface multi-modal on this path.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get('type') in ('input_text', 'output_text', 'text'):
            parts.append(block.get('text', '') or '')
    return '\n'.join(p for p in parts if p)


def _flatten_function_call_output(raw: Any) -> str:
    """``function_call_output.output`` may be a bare string OR a list of
    typed content parts (Responses spec allows both). Render to plain
    text so :class:`ChatMessageTool` can hold it."""
    if raw is None:
        return ''
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: List[str] = []
        for block in raw:
            if not isinstance(block, dict):
                continue
            if block.get('type') in ('input_text', 'output_text', 'text'):
                parts.append(block.get('text', '') or '')
        return '\n'.join(p for p in parts if p)
    return str(raw)


def _extract_reasoning_text(item: Dict[str, Any]) -> str:
    """Pull the textual content out of a ``reasoning`` input item.

    Supports both ``content[]`` (newer ``reasoning_text`` shape) and
    ``summary[]`` (``summary_text`` shape used by codex / o-series replay).
    Concatenates with newlines if both exist (rare in practice).
    """
    parts: List[str] = []
    for entry in item.get('content') or []:
        if isinstance(entry, dict) and entry.get('type') == 'reasoning_text':
            parts.append(entry.get('text', '') or '')
    for entry in item.get('summary') or []:
        if isinstance(entry, dict) and entry.get('type') == 'summary_text':
            parts.append(entry.get('text', '') or '')
    return '\n'.join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Request: tools[] / tool_choice
# ---------------------------------------------------------------------------


def responses_tools_to_tool_infos(tools: Sequence[Dict[str, Any]]) -> List[ToolInfo]:
    """Translate Responses ``tools[]`` into :class:`ToolInfo`.

    Unlike chat completions there is no ``function:`` wrapper layer — the
    function fields (``name`` / ``description`` / ``parameters``) live
    directly on the tool item.
    """
    out: List[ToolInfo] = []
    for spec in tools or []:
        if not isinstance(spec, dict):
            continue
        if spec.get('type') and spec.get('type') != 'function':
            continue
        name = spec.get('name')
        if not name:
            continue
        schema = spec.get('parameters') or {}
        params = _safe_tool_params(schema) if isinstance(schema, dict) else None
        out.append(
            ToolInfo(
                name=name,
                description=spec.get('description', '') or '',
                parameters=params or _safe_tool_params({}),
            )
        )
    return out


def responses_tool_choice(raw: Any) -> Optional[ToolChoice]:
    """Map Responses ``tool_choice`` — semantically identical to OpenAI chat
    completions on this path; reuse the chat helper verbatim."""
    return openai_tool_choice(raw)


# ---------------------------------------------------------------------------
# Response: ModelOutput → Responses payload (dict)
# ---------------------------------------------------------------------------


def model_output_to_responses_payload(
    output: ModelOutput,
    *,
    request_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a :class:`ModelOutput` as an OpenAI Responses payload dict.

    The dict is the value of the ``response`` field on
    ``response.created`` / ``response.completed`` SSE events, and also
    the JSON-mode response body. ``status`` is set to ``completed`` here;
    the SSE synthesizer overrides it to ``in_progress`` on the early
    frames.
    """
    output_items: List[Dict[str, Any]] = []

    if output.choices:
        msg = output.message
        text, reasoning = _split_text_and_reasoning(msg)

        if reasoning:
            output_items.append({
                'type': 'reasoning',
                'id': f'rs_{uuid.uuid4().hex[:24]}',
                'summary': [{
                    'type': 'summary_text',
                    'text': reasoning
                }],
            })
        if text:
            output_items.append({
                'type': 'message',
                'id': f'msg_{uuid.uuid4().hex[:24]}',
                'role': 'assistant',
                'content': [{
                    'type': 'output_text',
                    'text': text
                }],
                'status': 'completed',
            })
        for tc in msg.tool_calls or []:
            name, args = unpack_openai_tool_call(tc)
            output_items.append({
                'type': 'function_call',
                'id': f'fc_{uuid.uuid4().hex[:24]}',
                'call_id': tc.id,
                'name': name,
                'arguments': json.dumps(args, ensure_ascii=False),
                'status': 'completed',
            })

    usage = output.usage
    return {
        'id': output.id or f'resp_{uuid.uuid4().hex[:24]}',
        'object': 'response',
        'created_at': int(time.time()),
        'status': 'completed',
        'model': request_model or output.model or '',
        'output': output_items,
        'usage': {
            'input_tokens': usage.input_tokens if usage else 0,
            'output_tokens': usage.output_tokens if usage else 0,
            'total_tokens': usage.total_tokens if usage else 0,
        },
    }


# ---------------------------------------------------------------------------
# previous_response_id
# ---------------------------------------------------------------------------


def warn_unsupported_previous_response_id(body: Dict[str, Any]) -> None:
    """Stateful Responses sessions (``previous_response_id``) are not
    supported in P0 — the bridge always processes the full ``input[]``.
    codex ``exec`` occasionally sends this field; log a warning instead
    of returning 400 so the run continues."""
    pid = body.get('previous_response_id')
    if pid:
        logger.warning(
            f'bridge: /openai/v1/responses received previous_response_id={pid!r}; '
            f'stateful sessions not supported (P0), processing full input[] anyway'
        )


__all__ = [
    'model_output_to_responses_payload',
    'responses_request_to_messages',
    'responses_tool_choice',
    'responses_tools_to_tool_infos',
    'warn_unsupported_previous_response_id',
]
