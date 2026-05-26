"""ReAct agent strategy (function-calling mode).

Uses the model's native function-calling API but encourages step-by-step
reasoning through a specialised system prompt.  The loop terminates when
the model either (a) calls the ``submit`` tool with its final answer or
(b) replies without any tool calls.

Compared to the baseline ``FunctionCallingStrategy``, ReAct adds:

* A system prompt template that enforces THOUGHT → tool-call →
  OBSERVATION cycles.
* Automatic injection of the ``submit`` tool so the model can
  explicitly signal completion.
"""

from typing import List, Optional

# Import submit ToolInfo for auto-injection.
from evalscope.agent.tools.submit import SUBMIT_TOOL_INFO
from evalscope.api.agent import AgentContext, AgentLoopResult, AgentStrategy, ParsedAction, ToolSchemaMode
from evalscope.api.messages import ChatMessage, ChatMessageTool
from evalscope.api.model import ModelOutput
from evalscope.api.registry import register_strategy
from evalscope.api.tool import ToolCall, ToolCallError, ToolInfo

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _format_tools(tools: List[ToolInfo]) -> str:
    """Render a compact tool summary for the system prompt."""
    if not tools:
        return '(no tools available)'
    lines: list[str] = []
    for t in tools:
        arg_names = ', '.join(t.parameters.properties.keys()) if t.parameters else ''
        lines.append(f'- {t.name}({arg_names}): {t.description}')
    return '\n'.join(lines)


REACT_SYSTEM_PROMPT_TEMPLATE = """\
You are a reasoning-and-acting agent.  Solve the task step by step.

You have access to the following tools:
{tool_descriptions}

You MUST follow this format for every response:

1. **Think** – Briefly explain your reasoning about what to do next.
2. **Act**  – Call the appropriate tool (or the ``submit`` tool when you
   have the final answer).

Rules:
- Always explain your reasoning BEFORE making a tool call.
- After receiving an observation, reflect on it before taking the next step.
- When you are confident in the final answer, call the ``submit`` tool with
  your answer.  This immediately completes the task.
- If you are unsure, keep exploring with tool calls.
- Do NOT make up information.  Only use observations from tool calls.
"""


@register_strategy('react')
class ReactStrategy(AgentStrategy):
    """ReAct strategy that uses the model's native function-calling API with
    a reasoning-encouraging system prompt and ``submit`` tool support."""

    name: str = 'react'

    def __init__(self, *, system_prompt: Optional[str] = None, **_: object) -> None:
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------
    # AgentStrategy implementation
    # ------------------------------------------------------------------

    def build_system_prompt(self, ctx: AgentContext) -> Optional[str]:
        if self._system_prompt:
            return self._system_prompt
        return REACT_SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=_format_tools(ctx.tools), )

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

        # No tool calls at all – the model didn't follow the ReAct format.
        # Do NOT treat this as a final answer; the loop will inject a nudge
        # message so the model gets another chance to call submit.
        return ParsedAction(raw_text=message.text)

    def is_done(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        # Only done when submit was called (final_answer is set).
        # No-tool-call text is NOT a valid termination signal.
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
        # FC mode → standard ChatMessageTool.
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
        last assistant message's plain text content.
        """
        # Scan messages in reverse for a submit tool call.
        for msg in reversed(result.messages):
            if msg.role == 'assistant' and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function.name == 'submit':
                        answer = tc.function.arguments.get('answer', '')
                        if answer:
                            return str(answer)
        # Fallback: text-only content of the last model output. Using
        # ``.text`` (not ``str(content)``) so multimodal/reasoning
        # content parts don't leak their Python repr into the answer.
        return result.final_output.message.text or ''


__all__ = ['ReactStrategy']
