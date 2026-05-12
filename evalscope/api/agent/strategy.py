"""Strategy protocol: decides prompt shape, output parsing and termination.

A strategy does NOT execute tools and does NOT hold the environment.  It
only turns ``ModelOutput`` into a :class:`ParsedAction` and advises the
loop whether to stop.
"""

from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

from evalscope.api.messages import ChatMessage
from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolInfo
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


__all__ = ['AgentStrategy']
