"""Read the tool calls a local checkpoint writes as text.

A chat template that supports tools (Qwen and the templates modelled on it)
asks the model to answer with one block per call:

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Beijing"}}
    </tool_call>

The hosted-model path receives the same information as structured ``tool_calls``
from the API. A locally loaded checkpoint only returns the text, so the blocks
have to be read back before the result reaches a benchmark; otherwise a
tool-calling benchmark scores the model on an empty answer.
"""
import json
import re
from typing import List, Optional, Tuple

from evalscope.api.tool import ToolCall, ToolInfo, parse_tool_call

TOOL_CALL_BLOCK = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)


def parse_tool_calls_from_text(
    output: str, tools: Optional[List[ToolInfo]] = None
) -> Tuple[str, Optional[List[ToolCall]]]:
    """Split a model answer into its text and the tool calls it asked for.

    Args:
        output: The decoded model output.
        tools: The tools offered to the model, used to recover positional
            arguments that are not written as a JSON object.

    Returns:
        The answer with the parsed blocks removed, and the calls in the order
        they appeared (``None`` when the output holds no readable call). Blocks
        that do not parse are left in the text so nothing is lost silently.
    """
    if not output or '<tool_call>' not in output:
        return output, None

    calls: List[ToolCall] = []
    spans = []
    for match in TOOL_CALL_BLOCK.finditer(output):
        payload = _load_payload(match.group(1))
        if payload is None:
            continue
        name = payload.get('name')
        if not isinstance(name, str) or not name:
            continue
        arguments = payload.get('arguments', {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append(parse_tool_call(f'call_{len(calls)}', name, arguments, tools))
        spans.append(match.span())

    if not calls:
        return output, None

    content = output
    for start, end in reversed(spans):
        content = content[:start] + content[end:]
    return content.strip(), calls


def _load_payload(block: str) -> Optional[dict]:
    """Return the call payload, accepting both the flat and the nested form."""
    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    # Some templates wrap the call the way the OpenAI API returns it.
    function = payload.get('function')
    if isinstance(function, dict):
        return function
    return payload
