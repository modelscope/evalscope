"""Function-calling agent strategy.

Relies on the model's native function-calling output (``ToolCall`` entries
in ``ChatMessageAssistant.tool_calls``).  No custom prompt engineering or
text parsing is needed - the loop terminates as soon as the model returns
an assistant message without any tool calls, or calls the ``submit`` tool.
"""

from typing import List, Optional

# Import submit ToolInfo for auto-injection.
from evalscope.agent.tools.submit import SUBMIT_TOOL_INFO
from evalscope.api.agent import AgentContext, AgentLoopResult, AgentStrategy, ParsedAction, ToolSchemaMode
from evalscope.api.messages import ChatMessage, ChatMessageTool
from evalscope.api.model import ModelOutput
from evalscope.api.registry import register_strategy
from evalscope.api.tool import ToolCall, ToolCallError, ToolInfo


@register_strategy('function_calling')
class FunctionCallingStrategy(AgentStrategy):
    """Default strategy: delegate to the model's native tool-calling API.

    Supports the ``submit`` tool: when the model calls ``submit(answer=...)``
    the call is intercepted in :meth:`parse_output` and converted to a
    ``ParsedAction(final_answer=answer)`` so the loop terminates
    immediately without executing the tool.
    """

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

        # Intercept ``submit`` → treat as final answer.
        submit_calls = [tc for tc in tool_calls if tc.function.name == 'submit']
        if submit_calls:
            answer = submit_calls[0].function.arguments.get('answer', '')
            return ParsedAction(final_answer=answer, raw_text=message.text)

        if tool_calls:
            return ParsedAction(tool_calls=tool_calls, raw_text=message.text)

        # No tool calls – the model didn't use the submit tool.
        # Do NOT treat raw text as a final answer; the loop will inject
        # a nudge message so the model can try again.
        return ParsedAction(raw_text=message.text)

    def is_done(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        # Only done when submit was called (final_answer is set).
        return parsed.final_answer is not None

    def should_nudge(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        # Limit nudge to avoid token waste with models that prefer reasoning first.
        nudge_count = sum(1 for m in ctx.messages if m.role == 'user' and 'No tool was called' in str(m.content))
        return nudge_count < 2

    def tool_schema_mode(self) -> ToolSchemaMode:
        return 'function_calling'

    def tools(self, ctx: AgentContext) -> List[ToolInfo]:
        # Auto-inject the submit tool so the model can explicitly signal
        # completion regardless of the user's tool configuration.
        tool_list = list(ctx.tools)
        if not any(t.name == 'submit' for t in tool_list):
            tool_list.append(SUBMIT_TOOL_INFO)
        return tool_list

    def format_observation(
        self,
        call: ToolCall,
        observation: str,
        error: Optional[ToolCallError],
        parsed: ParsedAction,
        ctx: AgentContext,
    ) -> ChatMessage:
        """Format tool observations as :class:`ChatMessageTool` (FC default)."""
        return ChatMessageTool(
            content=observation,
            tool_call_id=call.id,
            function=call.function.name,
            error=error,
        )

    def extract_final_answer(self, result: AgentLoopResult) -> str:
        """Extract the final answer from the loop result.

        Prioritises the ``submit`` tool call's ``answer`` argument
        (the authoritative answer source), then falls back to the
        raw message content.
        """
        # Scan messages in reverse for a submit tool call.
        for msg in reversed(result.messages):
            if msg.role == 'assistant' and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function.name == 'submit':
                        answer = tc.function.arguments.get('answer', '')
                        if answer:
                            return str(answer)
        # Fallback: last model output content.
        return str(result.final_output.message.content or '')


__all__ = ['FunctionCallingStrategy']
