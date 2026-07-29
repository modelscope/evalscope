"""Shared :class:`AgentLoop` driver helper.

Centralises the async-coroutine wrapping + ``AsyncioLoopRunner`` invocation
+ ``finally runtime.close()`` boilerplate that is otherwise duplicated
between :meth:`DefaultDataAdapter._on_agent_inference` and
:meth:`AgentLoopAdapter._on_inference`.

The helper purposefully stops short of the final-answer extraction and
``ModelOutput`` post-processing so that callers retain full control over
their adapter-specific hooks (e.g.
:meth:`DefaultDataAdapter._extract_final_answer`).
"""

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage
from evalscope.api.model import Model
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner
from evalscope.utils.logger import get_logger
from .loop import AgentLoop
from .runtime import AgentRuntime
from .strategy import AgentStrategy
from .tool_executor import ToolExecutor, ToolHandler
from .trace import AgentTrace
from .types import AgentContext, AgentLoopResult

if TYPE_CHECKING:
    from .mcp import MCPServerConfig

logger = get_logger()


def run_agent_loop(
    *,
    model: Model,
    strategy: AgentStrategy,
    handlers: Dict[str, ToolHandler],
    runtime: Optional[AgentRuntime],
    initial_messages: List[ChatMessage],
    all_tools: List[Any],
    max_steps: int,
    sample_id: Optional[Any],
    trace_strategy_name: Optional[str],
    trace_runtime_name: Optional[str],
    mcp_configs: Optional[List['MCPServerConfig']] = None,
    close_runtime: bool = True,
) -> AgentLoopResult:
    """Drive a single :class:`AgentLoop` to completion and return its result.

    The runtime (when provided) is closed in a ``finally`` block by
    default so callers do not have to handle teardown themselves.
    ``AsyncioLoopRunner`` bridges the async loop into a synchronous call site.

    Args:
        model: The :class:`Model` driving generation.
        strategy: Pre-built :class:`AgentStrategy` instance.
        handlers: Mapping of tool name to :class:`ToolHandler` callable.
        runtime: Optional :class:`AgentRuntime`; closed on exit.
        initial_messages: Messages seeded into the :class:`AgentContext`.
        all_tools: Tool schemas (``ToolInfo``) advertised to the model.
        max_steps: Upper bound on loop iterations.
        sample_id: Identifier propagated into :class:`AgentContext`.
        trace_strategy_name: Strategy label recorded on :class:`AgentTrace`.
        trace_runtime_name: Agent runtime label recorded on :class:`AgentTrace`.
        mcp_configs: Optional list of MCP server configs whose advertised
            tools are merged into ``handlers`` / ``all_tools`` for the
            duration of the loop. Servers are spawned per sample (see
            :func:`evalscope.api.agent.mcp.resolve_mcp_tools`).
        close_runtime: Whether this helper owns and closes ``runtime``.
            Set to ``False`` when the caller needs to reuse the same
            runtime after the agent loop, for example to run a verifier.

    Returns:
        AgentLoopResult: Completed result with ``messages``, ``trace`` and
            ``final_output`` populated by the loop.
    """

    async def _run() -> AgentLoopResult:
        async with AsyncExitStack() as mcp_stack:
            merged_handlers: Dict[str, ToolHandler] = dict(handlers)
            merged_tools: List[Any] = list(all_tools)

            if mcp_configs:
                from .mcp import resolve_mcp_tools

                mcp_handler_map, mcp_tool_infos = await resolve_mcp_tools(mcp_configs, mcp_stack)
                for tool_name, handler in mcp_handler_map.items():
                    if tool_name in merged_handlers:
                        logger.warning(f'MCP tool {tool_name!r} shadows existing handler; last-write-wins')
                    merged_handlers[tool_name] = handler
                merged_tools.extend(mcp_tool_infos)

            try:
                tool_executor = ToolExecutor(handlers=merged_handlers, runtime=runtime)
                ctx = AgentContext(
                    sample_id=sample_id,
                    messages=initial_messages,
                    tools=merged_tools,
                    max_steps=max_steps,
                )
                trace = AgentTrace(
                    strategy=trace_strategy_name,
                    agent_runtime=trace_runtime_name,
                    max_steps=max_steps,
                )
                loop = AgentLoop(
                    model=model,
                    strategy=strategy,
                    tool_executor=tool_executor,
                    runtime=runtime,
                    max_steps=max_steps,
                    trace=trace,
                )
                return await loop.run(ctx)
            finally:
                if close_runtime and runtime is not None:
                    await runtime.close()

    return AsyncioLoopRunner.run(_run())
