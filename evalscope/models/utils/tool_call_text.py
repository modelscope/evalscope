"""Read the tool calls a local checkpoint writes as text.

Local chat templates ask the model to answer with tool calls written as text,
in a few common shapes:

- a Qwen/Hermes-style block per call:

     ```
     <tool_call>
     {"name": "get_weather", "arguments": {"city": "Beijing"}}
     ```

- a Mistral-style array prefix: ``[TOOL_CALLS] [{"name": ...}, ...]``

The hosted-model path receives the same information as structured ``tool_calls``
from the API. A locally loaded checkpoint only returns the text, so the calls
have to be read back before the result reaches a benchmark; otherwise a
tool-calling benchmark scores the model on an empty answer.

Templates differ in the payload key holding the values (``arguments`` vs
``parameters``); both are accepted, since silently dropping ``parameters``
produced valid-looking calls with empty arguments. Output that looks like a
tool call but cannot be parsed is kept as plain text (fail-closed) and logged
as a diagnostic; a plain-text answer with no call markers at all is left alone.
"""

import json
import re
from typing import Any, List, Optional, Tuple

from evalscope.api.tool import ToolCall, ToolInfo, parse_tool_call
from evalscope.utils.logger import get_logger

logger = get_logger()

TOOL_CALL_BLOCK = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
TOOL_CALLS_ARRAY = re.compile(r'\[TOOL_CALLS\]\s*(\[.*\])', re.DOTALL)
TOOL_CALL_MARKERS = ('<tool_call>', '[TOOL_CALLS]')

# A fixed message, emitted once, so a benchmark run full of unparseable calls
# does not flood the log with one warning per sample.
_DIAGNOSTIC_EMITTED = False


def parse_tool_calls_from_text(
    output: str, tools: Optional[List[ToolInfo]] = None
) -> Tuple[str, Optional[List[ToolCall]]]:
    """Split a model answer into its text and the tool calls it asked for.

    Args:
        output: The decoded model output.
        tools: The tools offered to the model, used to recover positional
            arguments that are not written as a JSON object.

    Returns:
        The answer with the parsed calls removed, and the calls in the order
        they appeared (``None`` when the output holds no readable call). Output
        that looks like a tool call but does not parse is logged and left in
        the text so nothing is lost silently.
    """
    if not output or not any(marker in output for marker in TOOL_CALL_MARKERS):
        return output, None

    calls: List[ToolCall] = []
    spans = []
    for match in TOOL_CALL_BLOCK.finditer(output):
        payload = _load_payload(match.group(1))
        if payload is None:
            continue
        call = _to_call(payload, len(calls), tools)
        if call is None:
            continue
        calls.append(call)
        spans.append(match.span())

    for match in TOOL_CALLS_ARRAY.finditer(output):
        if any(start < match.end() and match.start() < end for start, end in spans):
            continue
        added = 0
        for payload in _load_payloads(match.group(1)):
            call = _to_call(payload, len(calls), tools)
            if call is None:
                continue
            calls.append(call)
            added += 1
        if added:
            spans.append(match.span())

    if not calls:
        global _DIAGNOSTIC_EMITTED
        if not _DIAGNOSTIC_EMITTED:
            _DIAGNOSTIC_EMITTED = True
            logger.warning(
                'Output looks like a tool call but no call could be parsed; keeping it as plain text '
                '(this diagnostic is logged once).'
            )
        return output, None

    content = output
    for start, end in reversed(spans):
        content = content[:start] + content[end:]
    return content.strip(), calls


def _load_payload(block: str) -> Optional[dict]:
    """Return the call payload, accepting both the flat and the nested form."""
    payloads = _load_payloads(block)
    return payloads[0] if payloads else None


def _load_payloads(raw: str) -> List[dict]:
    """Return the call payloads in a JSON object or array, unwrapping the OpenAI-style form."""
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    result = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        # Some templates wrap the call the way the OpenAI API returns it.
        function = item.get('function')
        result.append(function if isinstance(function, dict) else item)
    return result


def _to_call(payload: dict, index: int, tools: Optional[List[ToolInfo]]) -> Optional[ToolCall]:
    """Build one call from a payload dict, accepting ``parameters`` as an alias of ``arguments``."""
    name = payload.get('name')
    if not isinstance(name, str) or not name:
        return None
    arguments = payload.get('arguments')
    if arguments is None:
        # Hermes-style templates write the values under ``parameters``; dropping them
        # silently produced valid-looking calls with empty arguments.
        arguments = payload.get('parameters', {})
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return parse_tool_call(f'call_{index}', name, arguments, tools)
