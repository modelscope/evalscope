"""SWE-bench backticks (textbased) agent strategy.

Aligns with the original ``mini-swe-agent`` ``swebench_backticks.yaml``
fallback configuration: the model emits exactly one shell command wrapped
in a ```` ```mswea_bash_command ``` ```` fenced code block; the strategy
parses it into a synthesized ``bash`` tool call.

Task completion is signalled by the literal sentinel
``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` printed on its own line in the
bash output.

This is **specifically** for SWE-bench evaluation against models that do
not support OpenAI-style function calling.  For general agentic tasks use
:class:`evalscope.agent.strategies.function_calling.FunctionCallingStrategy`.
"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional

from evalscope.api.agent import AgentContext, AgentLoopResult, AgentStrategy, ParsedAction, ToolSchemaMode
from evalscope.api.messages import ChatMessage, ChatMessageUser
from evalscope.api.model import ModelOutput
from evalscope.api.registry import register_strategy
from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction, ToolInfo
from ._observation import SUBMIT_SENTINEL, check_sentinel, format_exec_observation

# Regex for ``` ```mswea_bash_command ... ``` ``` fenced blocks (mirrors
# the original mini-swe-agent textbased action_regex).
_BASH_BLOCK_RE = re.compile(r'```mswea_bash_command\n(.*?)\n```', re.DOTALL)

# Minimal system prompt: SWE-bench business rules (workflow, submission
# protocol, modification boundary) are injected by the adapter as the first
# user (instance) message, not here.
SWE_BENCH_BACKTICKS_SYSTEM_PROMPT = (
    'You are a helpful assistant that can interact with a computer shell to solve programming tasks.'
)

_FORMAT_ERROR_TEMPLATE = (
    'Your response must contain exactly one '
    '```mswea_bash_command ... ``` fenced block with a single shell '
    'command. Do not emit multiple blocks.'
)


@register_strategy('swe_bench_backticks')
class SweBenchBackticksStrategy(AgentStrategy):
    """Textbased SWE-bench strategy mirroring ``swebench_backticks.yaml``.

    The model interleaves free-form THOUGHT prose with one fenced bash
    block per turn.  Output observations are returned as ``user`` messages
    (no function-calling tool channel) wrapped in the shared XML envelope.
    """

    name: str = 'swe_bench_backticks'

    def __init__(self, *, system_prompt: Optional[str] = None, **_: object) -> None:
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------
    # AgentStrategy interface
    # ------------------------------------------------------------------

    def build_system_prompt(self, ctx: AgentContext) -> Optional[str]:
        return self._system_prompt or SWE_BENCH_BACKTICKS_SYSTEM_PROMPT

    def prepare_messages(self, ctx: AgentContext) -> List[ChatMessage]:
        return ctx.messages

    def parse_output(self, output: ModelOutput, ctx: AgentContext) -> ParsedAction:
        content = output.message.text or ''
        bash_blocks = _BASH_BLOCK_RE.findall(content)

        if not bash_blocks:
            # No fenced block — return raw text and let ``should_nudge``
            # decide whether to retry.
            return ParsedAction(raw_text=content)

        if len(bash_blocks) > 1:
            # Original swebench_backticks.yaml requires exactly one block.
            return ParsedAction(
                raw_text=content,
                error=_FORMAT_ERROR_TEMPLATE,
            )

        command = bash_blocks[0].strip()
        call = ToolCall(
            id=f'sweb_{uuid.uuid4().hex[:8]}',
            function=ToolFunction(name='bash', arguments={'command': command}),
        )
        return ParsedAction(tool_calls=[call], raw_text=content)

    def is_done(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        # Single termination signal: ``parsed.final_answer`` set by
        # ``format_observation`` after detecting the completion sentinel
        # in the bash output.
        return parsed.final_answer is not None

    def should_nudge(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        # Allow one nudge per missing/format-error response.  Cap globally
        # at 2 so a buggy model doesn't burn the entire step budget on
        # nudges.
        marker = 'must contain exactly one'
        nudge_count = sum(1 for m in ctx.messages if m.role == 'user' and marker in str(m.content))
        return nudge_count < 2

    def tool_schema_mode(self) -> ToolSchemaMode:
        return 'textual_block'

    def tools(self, ctx: AgentContext) -> List[ToolInfo]:
        # Textual mode does not pass tool schemas to ``model.generate``.
        return []

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
            return ChatMessageUser(content=content)

        # Detect the completion sentinel BEFORE rendering any envelope —
        # mirrors mini-swe-agent ``DockerEnvironment._check_finished``.
        # On a hit we mutate ``parsed`` in place so the loop's single
        # ``is_done`` check terminates the run, and we archive the raw
        # submission payload (no XML envelope) so downstream patch
        # extraction never sees ``</output>``-style tags.
        submission = check_sentinel(observation)
        if submission is not None:
            parsed.final_answer = submission
            ctx.metadata['submission_source'] = 'sentinel'
            return ChatMessageUser(content=submission)

        content = format_exec_observation(observation)
        # Textbased models expect observations as user messages — they do
        # not interpret the OpenAI ``tool`` role.
        return ChatMessageUser(content=content)

    def extract_final_answer(self, result: AgentLoopResult) -> str:
        """Recover the sentinel-triggered submission from messages.

        The backticks strategy archives the sentinel payload as a user
        message whose content is the raw submission (no XML envelope).
        We scan in reverse for the most recent user message that is
        neither a ``<returncode>`` envelope nor a ``Tool call error``
        block — that's our submission payload.  Skip any user messages
        that don't look like envelopes either (e.g. the original task
        description) by stopping at the first non-system, non-assistant
        message after the last assistant turn.
        """
        # Walk backwards collecting user messages produced AFTER the last
        # assistant turn — those are tool observations.
        observations: list[str] = []
        for msg in reversed(result.messages):
            if msg.role == 'assistant':
                break
            if msg.role == 'user':
                observations.append(str(msg.content or ''))
        for content in observations:
            if not content:
                continue
            stripped = content.lstrip()
            if stripped.startswith('<returncode>'):
                continue
            if stripped.startswith('Tool call error'):
                continue
            return content
        return ''


__all__ = ['SUBMIT_SENTINEL', 'SweBenchBackticksStrategy']
