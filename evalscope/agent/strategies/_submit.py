"""Shared parsing for schema-validated ``submit`` tool calls."""

from typing import List, Optional

from evalscope.agent.tools.submit import SUBMIT_TOOL_INFO
from evalscope.api.agent import AgentContext, ParsedAction
from evalscope.api.tool import ToolCall, validate_tool_arguments


def parse_submit_action(
    tool_calls: List[ToolCall],
    raw_text: str,
    ctx: AgentContext,
    *,
    enabled: bool = True,
) -> Optional[ParsedAction]:
    """Return the terminal action for a submit call, or ``None`` if absent.

    Invalid submits remain executable actions when schema validation is enabled,
    allowing :class:`ToolExecutor` to return the standard parsing observation.
    """
    if not enabled:
        return None

    submit_call = next((call for call in tool_calls if call.function.name == 'submit'), None)
    if submit_call is None:
        return None

    if ctx.validate_tool_arguments and validate_tool_arguments(submit_call, SUBMIT_TOOL_INFO) is not None:
        return ParsedAction(tool_calls=[submit_call], raw_text=raw_text)
    return ParsedAction(final_answer=submit_call.function.arguments.get('answer', ''), raw_text=raw_text)
