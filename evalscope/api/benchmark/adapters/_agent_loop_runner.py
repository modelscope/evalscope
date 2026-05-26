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

import asyncio
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

logger = get_logger()


@dataclass
class _LoopMCPCache:
    """Per-event-loop MCP session cache.

    Each :class:`AsyncioLoopRunner`-owned loop carries one of these, holding
    an :class:`AsyncExitStack` that owns every spawned :class:`MCPServer`
    plus a config-keyed map of their advertised handlers / tool infos. The
    cache is torn down via a ``register_close_callback`` hook when the loop
    shuts down, so MCP subprocesses are released exactly once per worker
    thread instead of once per sample.
    """

    stack: AsyncExitStack
    # config-key -> (handlers, tool_infos, server_name)
    entries: Dict[str, Tuple[Dict[str, ToolHandler], List[Any], str]] = field(default_factory=dict)


_mcp_caches_lock = threading.Lock()
_mcp_caches: Dict[int, _LoopMCPCache] = {}


async def _resolve_mcp_tools(
    mcp_configs: List[Any],
    fallback_stack: AsyncExitStack,
) -> Tuple[Dict[str, ToolHandler], List[Any]]:
    """Spawn or look up MCP servers for the running loop and return their tools.

    When the coroutine runs on an :class:`AsyncioLoopRunner` loop, sessions
    are cached in a per-loop :class:`_LoopMCPCache` so 100 samples on the
    same worker thread share a single spawned subprocess (300+ stdio
    initialisations collapse to one per server). When the running loop is
    not runner-owned (e.g. a user calling ``run_agent_loop`` from
    ``asyncio.run``), sessions are entered into ``fallback_stack`` and torn
    down when the caller closes that stack — preserving the original
    per-call behaviour.
    """
    from evalscope.api.agent.mcp import MCPServer, mcp_tools

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    cache = await _get_or_create_loop_cache(loop) if loop is not None else None

    merged_handlers: Dict[str, ToolHandler] = {}
    merged_tool_infos: List[Any] = []

    for cfg in mcp_configs:
        if cache is not None:
            key = cfg.model_dump_json()
            cached_entry = cache.entries.get(key)
            if cached_entry is not None:
                handlers, infos, server_name = cached_entry
            else:
                server = MCPServer(cfg)
                await cache.stack.enter_async_context(server)
                handlers, infos = await mcp_tools(server)
                server_name = server.name
                cache.entries[key] = (handlers, infos, server_name)
        else:
            server = MCPServer(cfg)
            await fallback_stack.enter_async_context(server)
            handlers, infos = await mcp_tools(server)
            server_name = server.name

        for tool_name, handler in handlers.items():
            if tool_name in merged_handlers:
                logger.warning(
                    f'MCPServer[{server_name}]: tool {tool_name!r} shadows existing handler; last-write-wins'
                )
            merged_handlers[tool_name] = handler
        merged_tool_infos.extend(infos)

    return merged_handlers, merged_tool_infos


async def _get_or_create_loop_cache(loop: asyncio.AbstractEventLoop) -> Optional[_LoopMCPCache]:
    """Return the cache for ``loop`` (creating it on first call).

    Returns ``None`` when the loop is not one of :class:`AsyncioLoopRunner`'s
    own loops — caching requires a registrable close-callback hook to
    schedule subprocess teardown, and a foreign loop offers none.
    """
    loop_key = id(loop)
    with _mcp_caches_lock:
        existing = _mcp_caches.get(loop_key)
    if existing is not None:
        return existing

    new_cache = _LoopMCPCache(stack=AsyncExitStack())
    await new_cache.stack.__aenter__()

    async def _close() -> None:
        with _mcp_caches_lock:
            _mcp_caches.pop(loop_key, None)
        await new_cache.stack.aclose()

    if not AsyncioLoopRunner.register_close_callback(_close):
        # Foreign loop (e.g. plain ``asyncio.run``): give up on caching,
        # fall back to per-call lifecycle to avoid leaking subprocesses.
        await new_cache.stack.aclose()
        return None

    with _mcp_caches_lock:
        # A concurrent coroutine on the same loop may have raced ahead and
        # populated the cache while we awaited ``__aenter__`` above. Prefer
        # the winner so we don't end up with two parallel stacks.
        winner = _mcp_caches.setdefault(loop_key, new_cache)
    if winner is not new_cache:
        await new_cache.stack.aclose()
    return winner


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
    mcp_configs: Optional[List[Any]] = None,
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
            duration of the loop. Sessions are cached per worker-thread
            event loop (see :func:`_resolve_mcp_tools`), so the cost of
            spawning each server is paid once per thread — not once per
            sample.

    Returns:
        AgentLoopResult: Completed result with ``messages``, ``trace`` and
            ``final_output`` populated by the loop.
    """

    async def _run() -> AgentLoopResult:
        async with AsyncExitStack() as fallback_stack:
            merged_handlers: Dict[str, ToolHandler] = dict(handlers)
            merged_tools: List[Any] = list(all_tools)

            if mcp_configs:
                mcp_handler_map, mcp_tool_infos = await _resolve_mcp_tools(mcp_configs, fallback_stack)
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
