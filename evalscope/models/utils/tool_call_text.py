"""Read the tool calls a local checkpoint writes as text.

Local chat templates ask the model to answer with tool calls written as text,
in a few common shapes:

- a Qwen/Hermes-style block per call:

     <tool_call>
     {"name": "get_weather", "arguments": {"city": "Beijing"}}
     </tool_call>

- a Mistral-style array prefix: ``[TOOL_CALLS] [{"name": ...}, ...]``

The hosted-model path receives the same information as structured ``tool_calls``
from the API. A locally loaded checkpoint only returns the text, so the calls
have to be read back before the result reaches a benchmark; otherwise a
tool-calling benchmark scores the model on an empty answer.

Templates differ in the payload key holding the values (``arguments`` vs
``parameters``); both are accepted, since silently dropping ``parameters``
produced valid-looking calls with empty arguments. Mixed formats are read in
source order. An array containing any invalid element is rejected as a whole
and kept in the text, and output that looks like a tool call but cannot be
parsed is kept as plain text (fail-closed) and logged once as a diagnostic; a
plain-text answer with no call markers at all is left alone.
"""

import json
import re
from typing import Any, List, Optional, Tuple

from evalscope.api.tool import ToolCall, ToolInfo, parse_tool_call
from evalscope.utils.logger import get_logger

logger = get_logger()

TOOL_CALL_BLOCK = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
TOOL_CALLS_PREFIX = re.compile(r'\[TOOL_CALLS\]\s*')
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

    # Collect (position, payloads, span) per readable segment, then rebuild in
    # source order so mixed formats stay ordered and span removal stays valid.
    segments: List[Tuple[int, List[dict], Tuple[int, int]]] = []
    for match in TOOL_CALL_BLOCK.finditer(output):
        payload = _validated_payload(_load_json(match.group(1)))
        if payload is not None:
            segments.append((match.start(), [payload], match.span()))

    decoder = json.JSONDecoder()
    for match in TOOL_CALLS_PREFIX.finditer(output):
        try:
            value, end = decoder.raw_decode(output, match.end())
        except json.JSONDecodeError:
            continue
        if not isinstance(value, list):
            continue
        payloads = []
        for item in value:
            payload = _validated_payload(item)
            if payload is None:
                payloads = []  # one bad element fails the whole array (fail-closed)
                break
            payloads.append(payload)
        if payloads and not _overlaps(segments, match.start(), end):
            segments.append((match.start(), payloads, (match.start(), end)))

    if not segments:
        _warn_once()
        return output, None

    segments.sort(key=lambda segment: segment[0])

    calls: List[ToolCall] = []
    spans = []
    for _, payloads, span in segments:
        for payload in payloads:
            arguments = payload.get('arguments')
            if arguments is None:
                # Hermes-style templates write the values under ``parameters``.
                arguments = payload.get('parameters', {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append(parse_tool_call(f'call_{len(calls)}', payload['name'], arguments, tools))
        spans.append(span)

    content = output
    for start, end in reversed(spans):
        content = content[:start] + content[end:]
    return content.strip(), calls


def _warn_once() -> None:
    global _DIAGNOSTIC_EMITTED
    if not _DIAGNOSTIC_EMITTED:
        _DIAGNOSTIC_EMITTED = True
        logger.warning(
            'Output looks like a tool call but no call could be parsed; keeping it as plain text '
            '(this diagnostic is logged once).'
        )


def _overlaps(segments: List[Tuple[int, List[dict], Tuple[int, int]]], start: int, end: int) -> bool:
    """Whether a candidate span overlaps an already accepted segment span."""
    return any(a_start < end and start < a_end for _, _, (a_start, a_end) in segments)


def _load_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _validated_payload(item: Any) -> Optional[dict]:
    """Return a payload dict ready for call building, accepting the nested OpenAI-style form."""
    if not isinstance(item, dict):
        return None
    function = item.get('function')
    if isinstance(function, dict):
        item = function
    if not isinstance(item.get('name'), str) or not item.get('name'):
        return None
    return item
