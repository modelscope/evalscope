"""Function-calling agent strategy.

Relies on the model's native function-calling output (``ToolCall`` entries
in ``ChatMessageAssistant.tool_calls``).  No custom prompt engineering or
text parsing is needed - the loop terminates as soon as the model returns
an assistant message without any tool calls.
"""

from typing import List, Optional

from evalscope.api.agent import AgentContext, AgentLoopResult, AgentStrategy, ParsedAction, ToolSchemaMode
from evalscope.api.messages import ChatMessage
from evalscope.api.model import ModelOutput
from evalscope.api.registry import register_strategy
from evalscope.api.tool import ToolInfo


@register_strategy('function_calling')
class FunctionCallingStrategy(AgentStrategy):
    """Default strategy: delegate to the model's native tool-calling API."""

    name: str = 'function_calling'

    def __init__(self, *, system_prompt: Optional[str] = None, **_: object) -> None:
        self._system_prompt = system_prompt

    def build_system_prompt(self, ctx: AgentContext) -> Optional[str]:
        return self._system_prompt

    def prepare_messages(self, ctx: AgentContext) -> List[ChatMessage]:
        return ctx.messages

    def parse_output(self, output: ModelOutput, ctx: AgentContext) -> ParsedAction:
        message = output.message
        tool_calls = list(message.tool_calls or [])
        if tool_calls:
            return ParsedAction(tool_calls=tool_calls, raw_text=message.text)
        return ParsedAction(final_answer=message.text, raw_text=message.text)

    def is_done(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        # No pending tool calls → model has produced its final answer.
        return not parsed.tool_calls

    def tool_schema_mode(self) -> ToolSchemaMode:
        return 'function_calling'

    def tools(self, ctx: AgentContext) -> List[ToolInfo]:
        return list(ctx.tools)

    def extract_final_answer(self, result: AgentLoopResult) -> str:
        """Return the text content of the last model output.

        For function-calling strategy the loop always terminates on a
        tool-call-free assistant turn, so ``final_output.message.content``
        is guaranteed to be the model's direct answer.
        """
        return str(result.final_output.message.content or '')


__all__ = ['FunctionCallingStrategy']
