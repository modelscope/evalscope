"""SWE-bench toolcall agent strategy (mainline).

Aligns with the original ``mini-swe-agent`` ``swebench.yaml`` configuration:
the model is exposed a single ``bash`` tool via OpenAI function-calling and
signals task completion by emitting the literal sentinel
``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` followed by the final git patch in
its bash output (NOT by calling a ``submit`` tool).

When the sentinel is detected, ``format_observation`` raises
:class:`evalscope.api.agent.exceptions.Submitted` so the AgentLoop can
short-circuit just like mini-swe-agent's ``DockerEnvironment`` does.

This strategy is **not** a general-purpose strategy: do not register it as
the default ``function_calling`` replacement.  It is selected explicitly by
:class:`evalscope.benchmarks.swe_bench.SWEBenchAgenticAdapter`.
"""

from __future__ import annotations

from typing import List, Optional

from evalscope.api.agent import AgentContext, AgentLoopResult, AgentStrategy, ParsedAction, Submitted, ToolSchemaMode
from evalscope.api.messages import ChatMessage, ChatMessageTool
from evalscope.api.model import ModelOutput
from evalscope.api.registry import register_strategy
from evalscope.api.tool import ToolCall, ToolCallError, ToolInfo
from ._observation import SUBMIT_SENTINEL, check_sentinel, format_exec_observation


@register_strategy('swe_bench_toolcall')
class SweBenchToolcallStrategy(AgentStrategy):
    """SWE-bench toolcall strategy: bash + sentinel submission.

    Differences vs :class:`FunctionCallingStrategy`:
      * Does **not** auto-inject the ``submit`` tool — adapter must supply
        ``bash`` (and only ``bash``) so the sentinel protocol is the only
        completion path.
      * ``format_observation`` short-circuits the loop by raising
        :class:`Submitted` when the bash output starts with the
        completion sentinel.  ``is_done`` is therefore only a defensive
        fallback.
      * Observation envelope mirrors mini-swe-agent's
        ``observation_template`` (``<returncode>`` + ``<output>`` /
        head/tail truncation) for reliable model parsing.
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
        return 'You are a helpful assistant that can interact with a computer shell to solve programming tasks.'

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
        # The primary completion path is ``Submitted`` raised by
        # ``format_observation``.  Returning False here is safe because
        # the loop has already exited via the exception.
        return False

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
            return ChatMessageTool(
                content=content,
                tool_call_id=call.id,
                function=call.function.name,
                error=error,
            )

        # Detect the completion sentinel BEFORE rendering any envelope —
        # mirrors mini-swe-agent ``DockerEnvironment._check_finished``.
        submission = check_sentinel(observation)
        if submission is not None:
            raise Submitted(submission=submission)

        content = format_exec_observation(observation)
        return ChatMessageTool(
            content=content,
            tool_call_id=call.id,
            function=call.function.name,
            error=error,
        )

    def extract_final_answer(self, result: AgentLoopResult) -> str:
        """Return the sentinel-detected submission, if any."""
        submission = getattr(result, 'final_submission', None)
        if submission:
            return submission.strip()
        return ''


__all__ = [
    'SUBMIT_SENTINEL',
    'SweBenchToolcallStrategy',
]
