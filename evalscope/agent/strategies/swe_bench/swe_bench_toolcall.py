"""SWE-bench toolcall agent strategy (mainline).

Aligns with the original ``mini-swe-agent`` ``swebench.yaml`` configuration:
the model is exposed a single ``bash`` tool via OpenAI function-calling and
signals task completion by emitting the literal sentinel
``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` followed by the final git patch in
its bash output (NOT by calling a ``submit`` tool).

When the sentinel is detected, ``format_observation`` mutates the shared
``ParsedAction`` (sets ``parsed.final_answer``) so the AgentLoop's single
post-execution ``is_done(parsed, ctx)`` check breaks out of the loop.

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
from ._observation import SUBMIT_SENTINEL, check_sentinel, format_exec_observation


@register_strategy('swe_bench_toolcall')
class SweBenchToolcallStrategy(AgentStrategy):
    """SWE-bench toolcall strategy: bash + sentinel submission.

    Differences vs :class:`FunctionCallingStrategy`:
      * Does **not** auto-inject the ``submit`` tool — adapter must supply
        ``bash`` (and only ``bash``) so the sentinel protocol is the only
        completion path.
      * ``format_observation`` mutates the shared :class:`ParsedAction`
        by setting ``parsed.final_answer`` to the sentinel-detected
        submission payload.  The loop's post-execution ``is_done`` check
        then terminates without raising any exception.
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
        # Single termination signal: ``parsed.final_answer`` set either at
        # parse time (not used here — model-emitted submit calls are
        # rejected by ``parse_output``) or by ``format_observation`` after
        # detecting the completion sentinel in the bash output.
        return parsed.final_answer is not None

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
        parsed: ParsedAction,
        ctx: AgentContext,
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
        # On a hit we mutate ``parsed`` in place so the loop's single
        # ``is_done`` check terminates the run, and we archive the raw
        # submission payload (no XML envelope) into ``ctx.messages`` so
        # downstream patch extraction never sees ``</output>``-style tags.
        submission = check_sentinel(observation)
        if submission is not None:
            parsed.final_answer = submission
            ctx.metadata['submission_source'] = 'sentinel'
            return ChatMessageTool(
                content=submission,
                tool_call_id=call.id,
                function=call.function.name,
            )

        content = format_exec_observation(observation)
        return ChatMessageTool(
            content=content,
            tool_call_id=call.id,
            function=call.function.name,
            error=error,
        )

    def extract_final_answer(self, result: AgentLoopResult) -> str:
        """Recover the sentinel-triggered submission from messages.

        ``format_observation`` archives the sentinel payload as a tool
        message whose content is the raw submission (no XML envelope).
        We scan in reverse for the most recent tool message that is
        neither a ``<returncode>`` envelope nor a ``Tool call error``
        block — that's our submission payload.
        """
        for msg in reversed(result.messages):
            if msg.role != 'tool':
                continue
            content = str(msg.content or '')
            if not content:
                continue
            stripped = content.lstrip()
            if stripped.startswith('<returncode>'):
                continue
            if stripped.startswith('Tool call error'):
                continue
            return content
        return ''


__all__ = [
    'SUBMIT_SENTINEL',
    'SweBenchToolcallStrategy',
]
