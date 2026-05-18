"""Strategy protocol: decides prompt shape, output parsing and termination.

A strategy does NOT execute tools and does NOT hold the environment.  It
only turns ``ModelOutput`` into a :class:`ParsedAction` and advises the
loop whether to stop.

Termination contract
--------------------
The AgentLoop has a **single** termination signal: ``ParsedAction.final_answer``.
A strategy may set it from two places:

1. ``parse_output()``  — when the model output itself encodes completion
   (e.g. ``submit(answer=...)`` tool call).  No tool execution happens.
2. ``format_observation()`` — when completion is encoded inside a tool
   result (e.g. SWE-bench's ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
   sentinel printed by bash).  The strategy mutates the ``parsed``
   argument in place.

Either way, the loop checks ``is_done(parsed, ctx)`` after the relevant
phase and breaks out.  No exceptions are used for control flow.
"""

from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

from evalscope.api.messages import ChatMessage, ChatMessageTool
from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolCall, ToolCallError, ToolInfo
from .types import AgentContext, AgentLoopResult, ParsedAction, ToolSchemaMode


@runtime_checkable
class AgentStrategy(Protocol):
    """Pluggable policy driving the AgentLoop.

    Implementations live under ``evalscope/agent/strategies/`` and are
    registered via ``@register_strategy('name')``.
    """

    name: str
    """Registered strategy name, used for trace labeling."""

    def build_system_prompt(self, ctx: AgentContext) -> Optional[str]:
        """Return the system prompt injected at step 0, or None to skip."""
        ...

    def prepare_messages(self, ctx: AgentContext) -> List[ChatMessage]:
        """Return the message list passed to ``model.generate`` this turn.

        Default implementation may simply return ``ctx.messages``.  Strategies
        like ReAct can inject extra scratchpad formatting here.
        """
        ...

    def parse_output(self, output: ModelOutput, ctx: AgentContext) -> ParsedAction:
        """Extract tool calls / final answer / errors from model output."""
        ...

    def is_done(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        """Return True to terminate the loop immediately after this step."""
        ...

    def tool_schema_mode(self) -> ToolSchemaMode:
        """How should the loop surface tools to ``model.generate``?"""
        ...

    def tools(self, ctx: AgentContext) -> List[ToolInfo]:
        """Tools to pass to ``model.generate`` when
        ``tool_schema_mode() == 'function_calling'``.  Default: ``ctx.tools``.
        """
        ...

    def extract_final_answer(self, result: AgentLoopResult) -> str:
        """Extract the final prediction string from a completed loop run.

        Called by :meth:`DefaultDataAdapter._extract_final_answer` after the
        loop finishes.  The default for ``function_calling`` is the content of
        the last ``model.generate`` output.  Other strategies (e.g. ReAct) may
        scan the message history for a specific tag or tool argument.

        Benchmarks that need custom extraction should override
        :meth:`DefaultDataAdapter._extract_final_answer` instead of this
        method.
        """
        ...

    def should_nudge(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        """Whether to inject a nudge when no tool_calls are produced.

        Override in subclasses to customize nudge behavior.  Return False to
        skip the nudge and treat the current output as an implicit final answer.
        """
        ...

    def format_observation(
        self,
        call: ToolCall,
        observation: str,
        error: Optional[ToolCallError],
        parsed: ParsedAction,
        ctx: AgentContext,
    ) -> ChatMessage:
        """Format a tool execution result as a :class:`ChatMessage`.

        The default implementation returns a :class:`ChatMessageTool`.
        Textual strategies (e.g. ``swe_bench_backticks``) override this to return a
        :class:`ChatMessageUser` with strategy-specific formatting so that
        models which do not use function-calling can understand observations.

        A strategy MAY also signal task completion here by mutating the
        ``parsed`` argument in place (typically by setting
        ``parsed.final_answer``).  The loop will detect this through its
        post-execution ``is_done(parsed, ctx)`` check and terminate
        without raising any exception.  The returned ``ChatMessage`` is
        appended to ``ctx.messages`` as-is — strategies wishing to archive
        a "clean" submission payload (without any XML envelope) should
        return a message whose content is exactly that payload.
        """
        ...


__all__ = ['AgentStrategy']
