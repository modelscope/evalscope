"""Google Gemini API ⇄ EvalScope native type translation.

Translates the Gemini ``generateContent`` request/response wire format
into EvalScope's ``ChatMessage`` / ``ModelOutput`` types.  The bridge
exposes ``/gemini/v1beta/models/{model}:generateContent`` and
``/gemini/v1beta/models/{model}:streamGenerateContent`` routes so
Gemini CLI (which speaks the native Google AI protocol via
``GOOGLE_GEMINI_BASE_URL``) can talk to any EvalScope-managed model.

Reference protocol: https://ai.google.dev/api/generate-content
Reference implementation: inspect_ai/agent/_bridge/google_api_impl.py
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evalscope.api.messages import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.api.tool import ToolCall, ToolChoice, ToolFunction, ToolInfo, ToolParams

# -- Request → ChatMessage translation --


def gemini_request_to_messages(body: Dict[str, Any]) -> List[ChatMessage]:
    """Convert a Gemini ``generateContent`` request body to ``ChatMessage[]``.

    Gemini wire format uses:
    - ``systemInstruction`` (top-level, outside ``contents``)
    - ``contents[]`` with ``role`` = ``"user"`` | ``"model"``
    - ``parts[]`` within each content entry (text, functionCall, functionResponse, inlineData)
    """
    messages: List[ChatMessage] = []

    # System instruction (out-of-band, similar to Anthropic)
    system_instruction = body.get('systemInstruction') or body.get('system_instruction')
    if system_instruction:
        sys_text = _extract_system_text(system_instruction)
        if sys_text:
            messages.append(ChatMessageSystem(content=sys_text))

    # Track tool call IDs by function name for matching with functionResponse
    pending_tool_calls: Dict[str, List[str]] = {}

    for content in body.get('contents') or []:
        if not isinstance(content, dict):
            continue
        role = content.get('role', 'user')
        parts = content.get('parts') or []

        if role == 'user':
            user_content, tool_messages = _extract_user_parts(parts, pending_tool_calls)
            # Tool results first, then user text (matches EvalScope ordering)
            messages.extend(tool_messages)
            if user_content:
                messages.append(ChatMessageUser(content=user_content))

        elif role == 'model':
            text, tool_calls = _extract_model_parts(parts)
            # Update pending_tool_calls for next user turn's functionResponse matching
            pending_tool_calls.clear()
            for tc in tool_calls:
                fn_name = tc.function.name if isinstance(tc.function, ToolFunction) else str(tc.function)
                if fn_name not in pending_tool_calls:
                    pending_tool_calls[fn_name] = []
                pending_tool_calls[fn_name].append(tc.id)

            messages.append(ChatMessageAssistant(
                content=text or '',
                tool_calls=tool_calls or None,
            ))

    return messages


def _extract_system_text(system_instruction: Any) -> str:
    """Extract text from systemInstruction (can be dict with parts or list)."""
    if isinstance(system_instruction, str):
        return system_instruction
    if isinstance(system_instruction, dict):
        parts = system_instruction.get('parts') or []
        return _text_from_parts(parts)
    if isinstance(system_instruction, list):
        texts = []
        for item in system_instruction:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
        return '\n\n'.join(texts)
    return ''


def _text_from_parts(parts: List[Any]) -> str:
    """Concatenate text parts."""
    texts = []
    for part in parts:
        if isinstance(part, dict) and 'text' in part:
            texts.append(part['text'])
        elif isinstance(part, str):
            texts.append(part)
    return ''.join(texts)


def _extract_user_parts(
    parts: List[Any],
    pending_tool_calls: Dict[str, List[str]],
) -> Tuple[str, List[ChatMessageTool]]:
    """Extract user text and functionResponse tool messages from parts."""
    text_parts: List[str] = []
    tool_messages: List[ChatMessageTool] = []

    for part in parts:
        if not isinstance(part, dict):
            if isinstance(part, str):
                text_parts.append(part)
            continue

        if 'text' in part:
            text_parts.append(part['text'])

        elif 'functionResponse' in part or 'function_response' in part:
            func_response = part.get('functionResponse') or part.get('function_response') or {}
            func_name = func_response.get('name', '')
            response = func_response.get('response', {})

            # Match with pending tool call ID
            if func_name in pending_tool_calls and pending_tool_calls[func_name]:
                call_id = pending_tool_calls[func_name].pop(0)
            else:
                call_id = f'call_{func_name}_{uuid.uuid4().hex[:8]}'

            response_content = json.dumps(response) if isinstance(response, dict) else str(response)
            tool_messages.append(ChatMessageTool(
                content=response_content,
                tool_call_id=call_id,
                function=func_name,
            ))

    user_text = '\n'.join(p for p in text_parts if p)
    return user_text, tool_messages


def _extract_model_parts(parts: List[Any]) -> Tuple[str, List[ToolCall]]:
    """Extract assistant text and functionCall tool calls from model parts."""
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []

    for part_idx, part in enumerate(parts):
        if not isinstance(part, dict):
            if isinstance(part, str):
                text_parts.append(part)
            continue

        if 'text' in part:
            text_parts.append(part['text'])

        elif 'functionCall' in part or 'function_call' in part:
            func_call = part.get('functionCall') or part.get('function_call') or {}
            func_name = func_call.get('name', '')
            args = func_call.get('args') or {}
            if not isinstance(args, dict):
                args = {'value': args}

            # Deterministic call ID for stability
            call_id = f'call_{func_name}_{uuid.uuid4().hex[:8]}'
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    function=ToolFunction(name=func_name, arguments=args),
                    type='function',
                )
            )

    text = '\n'.join(p for p in text_parts if p)
    return text, tool_calls


# -- Tool translation --


def gemini_tools_to_tool_infos(tools: Sequence[Dict[str, Any]]) -> List[ToolInfo]:
    """Translate Gemini ``tools[]`` (with ``functionDeclarations``) to ``ToolInfo``."""
    out: List[ToolInfo] = []
    for google_tool in tools or []:
        if not isinstance(google_tool, dict):
            continue
        if 'functionDeclarations' in google_tool:
            for func_decl in google_tool['functionDeclarations']:
                parameters = func_decl.get('parameters') or func_decl.get('parametersJsonSchema') or {}
                params = _safe_tool_params(parameters) if isinstance(parameters, dict) else ToolParams()
                out.append(
                    ToolInfo(
                        name=func_decl.get('name', ''),
                        description=func_decl.get('description', ''),
                        parameters=params,
                    )
                )
    return out


def gemini_tool_choice(tool_config: Any) -> Optional[ToolChoice]:
    """Map Gemini ``toolConfig.functionCallingConfig`` to EvalScope ToolChoice."""
    if not tool_config or not isinstance(tool_config, dict):
        return None
    fcc = tool_config.get('functionCallingConfig') or tool_config.get('function_calling_config') or {}
    mode = fcc.get('mode', 'AUTO')
    if mode == 'AUTO':
        return 'auto'
    elif mode == 'ANY':
        return 'any'
    elif mode == 'NONE':
        return 'none'
    else:
        allowed = fcc.get('allowedFunctionNames') or []
        if allowed and len(allowed) == 1:
            return ToolFunction(name=allowed[0], arguments={})
        return 'auto'


def _safe_tool_params(schema: Dict[str, Any]) -> ToolParams:
    try:
        return ToolParams.model_validate({
            'properties': schema.get('properties', {}) or {},
            'required': schema.get('required', []) or [],
        })
    except Exception:
        return ToolParams()


# -- ModelOutput → Gemini response translation --


def model_output_to_gemini_response(
    output: ModelOutput,
    *,
    request_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a :class:`ModelOutput` as a Gemini ``generateContent`` response."""
    parts: List[Dict[str, Any]] = []

    message = output.message if output.choices else None
    if message is not None:
        # Text content
        text = message.text or ''
        if text:
            parts.append({'text': text})

        # Function calls
        for tc in message.tool_calls or []:
            name, args = _unpack_gemini_tool_call(tc)
            parts.append({'functionCall': {'name': name, 'args': args}})

    if not parts:
        parts.append({'text': ''})

    stop_reason = output.choices[0].stop_reason if output.choices else 'stop'
    usage = output.usage

    response: Dict[str, Any] = {
        'candidates': [{
            'content': {
                'parts': parts,
                'role': 'model'
            },
            'finishReason': map_stop_reason_to_gemini(stop_reason),
            'index': 0,
            'safetyRatings': [],
        }],
        'usageMetadata': {
            'promptTokenCount': usage.input_tokens if usage else 0,
            'candidatesTokenCount': usage.output_tokens if usage else 0,
            'totalTokenCount': usage.total_tokens if usage else 0,
        },
        'modelVersion': request_model or output.model or '',
    }
    return response


def _unpack_gemini_tool_call(tool_call: ToolCall) -> Tuple[str, Dict[str, Any]]:
    """Return ``(name, arguments)`` for a ToolCall."""
    fn = tool_call.function
    if isinstance(fn, ToolFunction):
        return fn.name, fn.arguments or {}
    return str(fn), {}


# -- Stop reason mapping --

_STOP_REASON_MAP = {
    'stop': 'STOP',
    'tool_calls': 'STOP',
    'max_tokens': 'MAX_TOKENS',
    'model_length': 'MAX_TOKENS',
    'content_filter': 'SAFETY',
    'unknown': 'STOP',
}


def map_stop_reason_to_gemini(reason: str) -> str:
    """Translate an EvalScope ``StopReason`` to Gemini ``finishReason``."""
    return _STOP_REASON_MAP.get(reason, 'STOP')


# -- Generate config extraction --


def build_gemini_generate_config(body: Dict[str, Any]) -> GenerateConfig:
    """Extract generation config from a Gemini request body."""
    gen_cfg = body.get('generationConfig') or body.get('generation_config') or {}
    kwargs: Dict[str, Any] = {'stream': True}
    if 'maxOutputTokens' in gen_cfg or 'max_output_tokens' in gen_cfg:
        kwargs['max_tokens'] = gen_cfg.get('maxOutputTokens') or gen_cfg.get('max_output_tokens')
    if 'temperature' in gen_cfg:
        kwargs['temperature'] = gen_cfg['temperature']
    if 'topP' in gen_cfg or 'top_p' in gen_cfg:
        kwargs['top_p'] = gen_cfg.get('topP') or gen_cfg.get('top_p')
    stop_seqs = gen_cfg.get('stopSequences') or gen_cfg.get('stop_sequences')
    if stop_seqs:
        kwargs['stop_seqs'] = list(stop_seqs)
    return GenerateConfig(**kwargs)


# -- Path utilities --


def extract_model_from_path(path: str) -> str:
    """Extract model name from Google API path.

    Examples:
        /gemini/v1beta/models/gemini-2.5-pro:generateContent -> gemini-2.5-pro
        /gemini/models/gemini-2.5-flash:streamGenerateContent -> gemini-2.5-flash
    """
    match = re.search(r'models/([^/:]+)', path)
    return match.group(1) if match else 'gemini'


def is_streaming_path(path: str) -> bool:
    """Check if the path is a streaming generateContent request."""
    return ':streamGenerateContent' in path
