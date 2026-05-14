"""SWE-bench toolcall agent strategy (mainline).

Aligns with the original ``mini-swe-agent`` ``swebench.yaml`` configuration:
the model is exposed a single ``bash`` tool via OpenAI function-calling and
signals task completion by emitting the literal sentinel
``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` followed by the final git patch in
its bash output (NOT by calling a ``submit`` tool).

The post-execution ``is_done`` hook scans the most recent tool message for
the sentinel; ``extract_final_answer`` returns the patch text appearing
*after* the sentinel.

This strategy is **not** a general-purpose strategy: do not register it as
the default ``function_calling`` replacement.  It is selected explicitly by
:class:`evalscope.benchmarks.swe_bench.SWEBenchAgenticAdapter`.
"""

from __future__ import annotations

from typing import List, Optional

from evalscope.api.agent import AgentContext, AgentLoopResult, AgentStrategy, ParsedAction, ToolSchemaMode
from evalscope.api.messages import ChatMessage, ChatMessageTool
from evalscope.api.model import ModelOutput
from evalscope.api.registry import register_strategy
from evalscope.api.tool import ToolCall, ToolCallError, ToolInfo
from ._observation import format_exec_observation

# Literal sentinel printed by the model on its own line to submit a patch.
SUBMIT_SENTINEL = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'


def _sentinel_payload(content: str) -> Optional[str]:
    """Return the patch text following the sentinel line, or None.

    The sentinel must appear as its own (stripped) line — substring matches
    inside arbitrary prose / prompts are ignored.  Everything from the line
    *after* the sentinel up to the end of the message is treated as the
    submission payload.
    """
    if not content:
        return None
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == SUBMIT_SENTINEL:
            payload = '\n'.join(lines[idx + 1:])
            return payload
    return None


@register_strategy('swe_bench_toolcall')
class SweBenchToolcallStrategy(AgentStrategy):
    """SWE-bench toolcall strategy: bash + sentinel submission.

    Differences vs :class:`FunctionCallingStrategy`:
      * Does **not** auto-inject the ``submit`` tool — adapter must supply
        ``bash`` (and only ``bash``) so the sentinel protocol is the only
        completion path.
      * ``is_done`` checks the latest tool message for the
        ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` sentinel.
      * ``format_observation`` wraps the bash observation in the XML
        envelope used by the original ``swebench.yaml`` (``<output>...
        </output>`` with head/tail truncation).
    """

    name: str = 'swe_bench_toolcall'

    def __init__(self, *, system_prompt: Optional[str] = None, **_: object) -> None:
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------
    # AgentStrategy interface
    # ------------------------------------------------------------------

    def build_system_prompt(self, ctx: AgentContext) -> Optional[str]:
        # Original swebench.yaml uses a single short line; SWE-bench business
        # rules (workflow / submission protocol) are injected as the user
        # instance_template by the adapter, not here.
        if self._system_prompt is not None:
            return self._system_prompt
        return 'You are a helpful assistant that can interact with a computer shell.'

    def prepare_messages(self, ctx: AgentContext) -> List[ChatMessage]:
        return ctx.messages

    def parse_output(self, output: ModelOutput, ctx: AgentContext) -> ParsedAction:
        message = output.message
        tool_calls = list(message.tool_calls or [])

        # Only ``bash`` is allowed; any other tool name is rejected so the
        # adapter doesn't silently fall back to ``submit`` semantics.
        bash_calls = [tc for tc in tool_calls if tc.function.name == 'bash']
        if bash_calls:
            return ParsedAction(tool_calls=bash_calls, raw_text=message.text)

        # No bash tool calls — let the loop nudge once.
        return ParsedAction(raw_text=message.text)

    def is_done(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        # Pre-execution: nothing terminates here in toolcall mode (no
        # ``submit`` shortcut).  Post-execution: scan the last tool message.
        if not ctx.messages:
            return False
        last = ctx.messages[-1]
        if last.role != 'tool':
            return False
        return _sentinel_payload(str(last.content or '')) is not None

    def should_nudge(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        # Allow at most one nudge; matches the ``FunctionCallingStrategy``
        # philosophy but with a distinct marker so the two don't share state.
        marker = 'No bash tool was called'
        nudge_count = sum(1 for m in ctx.messages if m.role == 'user' and marker in str(m.content))
        return nudge_count < 1

    def tool_schema_mode(self) -> ToolSchemaMode:
        return 'function_calling'

    def tools(self, ctx: AgentContext) -> List[ToolInfo]:
        # Adapter is responsible for providing the ``bash`` tool; do not
        # auto-inject ``submit`` (sentinel protocol is the only completion).
        return list(ctx.tools)

    def format_observation(
        self,
        call: ToolCall,
        observation: str,
        error: Optional[ToolCallError],
    ) -> ChatMessage:
        if error is not None:
            content = format_exec_observation('', error_message=error.message)
        else:
            content = format_exec_observation(observation)
        return ChatMessageTool(
            content=content,
            tool_call_id=call.id,
            function=call.function.name,
            error=error,
        )

    def extract_final_answer(self, result: AgentLoopResult) -> str:
        """Return patch text following the most recent sentinel marker."""
        for msg in reversed(result.messages):
            if msg.role == 'tool':
                payload = _sentinel_payload(str(msg.content or ''))
                if payload is not None:
                    return payload.strip()
        # Fallback: empty string lets the adapter run ``extract_diff`` on the
        # last assistant content as a best-effort recovery.
        return ''


__all__ = ['SweBenchToolcallStrategy', 'SUBMIT_SENTINEL']
