"""Shared :class:`AgentLoop` driver helper.

Centralises the async-coroutine wrapping + ``AsyncioLoopRunner`` invocation
+ ``finally environment.close()`` boilerplate that is otherwise duplicated
between :meth:`DefaultDataAdapter._on_agent_inference` and
:meth:`AgentLoopAdapter._on_inference`.

The helper purposefully stops short of the final-answer extraction and
``ModelOutput`` post-processing so that callers retain full control over
their adapter-specific hooks (e.g.
:meth:`DefaultDataAdapter._extract_final_answer`).
"""

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from evalscope.api.agent import (
    AgentContext,
    AgentEnvironment,
    AgentLoop,
    AgentLoopResult,
    AgentStrategy,
    AgentTrace,
    ToolExecutor,
    ToolHandler,
)
from evalscope.api.messages import ChatMessage
from evalscope.api.model import Model
from evalscope.utils.function_utils import AsyncioLoopRunner
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.agent.mcp import MCPServerConfig

logger = get_logger()


def run_agent_loop(
    *,
    model: Model,
    strategy: AgentStrategy,
    handlers: Dict[str, ToolHandler],
    environment: Optional[AgentEnvironment],
    initial_messages: List[ChatMessage],
    all_tools: List[Any],
    max_steps: int,
    sample_id: Optional[Any],
    trace_strategy_name: Optional[str],
    trace_env_name: Optional[str],
    mcp_configs: Optional[List['MCPServerConfig']] = None,
) -> AgentLoopResult:
    """Drive a single :class:`AgentLoop` to completion and return its result.

    The environment (when provided) is closed in a ``finally`` block so
    callers do not have to handle teardown themselves. ``AsyncioLoopRunner``
    bridges the async loop into a synchronous call site.

    Args:
        model: The :class:`Model` driving generation.
        strategy: Pre-built :class:`AgentStrategy` instance.
        handlers: Mapping of tool name to :class:`ToolHandler` callable.
        environment: Optional :class:`AgentEnvironment`; closed on exit.
        initial_messages: Messages seeded into the :class:`AgentContext`.
        all_tools: Tool schemas (``ToolInfo``) advertised to the model.
        max_steps: Upper bound on loop iterations.
        sample_id: Identifier propagated into :class:`AgentContext`.
        trace_strategy_name: Strategy label recorded on :class:`AgentTrace`.
        trace_env_name: Environment label recorded on :class:`AgentTrace`.
        mcp_configs: Optional list of MCP server configs whose advertised
            tools are merged into ``handlers`` / ``all_tools`` for the
            duration of the loop. Servers are spawned per sample (see
            :func:`evalscope.api.agent.mcp.resolve_mcp_tools`).

    Returns:
        AgentLoopResult: Completed result with ``messages``, ``trace`` and
            ``final_output`` populated by the loop.
    """

    async def _run() -> AgentLoopResult:
        async with AsyncExitStack() as mcp_stack:
            merged_handlers: Dict[str, ToolHandler] = dict(handlers)
            merged_tools: List[Any] = list(all_tools)

            if mcp_configs:
                from evalscope.api.agent.mcp import resolve_mcp_tools

                mcp_handler_map, mcp_tool_infos = await resolve_mcp_tools(mcp_configs, mcp_stack)
                for tool_name, handler in mcp_handler_map.items():
                    if tool_name in merged_handlers:
                        logger.warning(f'MCP tool {tool_name!r} shadows existing handler; last-write-wins')
                    merged_handlers[tool_name] = handler
                merged_tools.extend(mcp_tool_infos)

            try:
                tool_executor = ToolExecutor(handlers=merged_handlers, environment=environment)
                ctx = AgentContext(
                    sample_id=sample_id,
                    messages=initial_messages,
                    tools=merged_tools,
                    max_steps=max_steps,
                )
                trace = AgentTrace(
                    strategy=trace_strategy_name,
                    environment=trace_env_name,
                    max_steps=max_steps,
                )
                loop = AgentLoop(
                    model=model,
                    strategy=strategy,
                    tool_executor=tool_executor,
                    environment=environment,
                    max_steps=max_steps,
                    trace=trace,
                )
                return await loop.run(ctx)
            finally:
                if environment is not None:
                    await environment.close()

    return AsyncioLoopRunner.run(_run())
