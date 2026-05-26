"""Adapt MCP server tools into EvalScope's :class:`ToolHandler` + :class:`ToolInfo` shapes.

Bridges :class:`MCPServer` (which speaks the official ``mcp`` SDK shapes)
to the same ``handlers: Dict[str, ToolHandler]`` and ``all_tools: List[ToolInfo]``
that ``AgentLoop`` already consumes — so MCP tools are indistinguishable
from native ones (``bash`` / ``python_exec`` / ...) from the loop's POV.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.tool import ToolCall, ToolInfo, ToolParams
from evalscope.utils.json_schema import JSONSchema
from evalscope.utils.logger import get_logger
from .client import MCPServer
from .types import MCPServerConfig

logger = get_logger()


def _tool_matches_filter(name: str, filter_spec: Any) -> bool:
    if filter_spec == 'all':
        return True
    if isinstance(filter_spec, list):
        return name in filter_spec
    return False


def _to_tool_info(mcp_tool: Any) -> ToolInfo:
    """Convert an ``mcp.types.Tool`` to evalscope's :class:`ToolInfo`.

    MCP tools advertise their input schema via ``inputSchema`` (raw JSON
    Schema dict).  We coerce it through :class:`JSONSchema` and pack into
    :class:`ToolParams`; missing / malformed schemas degrade to the
    no-argument default.
    """
    schema = getattr(mcp_tool, 'inputSchema', None) or {}
    properties_dict: Dict[str, Any] = schema.get('properties') or {}
    required_list: List[str] = list(schema.get('required') or [])
    additional = schema.get('additionalProperties', False)

    properties: Dict[str, JSONSchema] = {}
    for prop_name, prop_schema in properties_dict.items():
        if isinstance(prop_schema, dict):
            try:
                properties[prop_name] = JSONSchema.model_validate(prop_schema)
            except Exception as ex:  # pragma: no cover - defensive
                logger.debug(
                    f'MCP tool {mcp_tool.name!r}: dropping unparseable property '
                    f'{prop_name!r} ({ex}); falling back to free-form string'
                )
                properties[prop_name] = JSONSchema(type='string')

    params = ToolParams(
        properties=properties,
        required=required_list,
        additionalProperties=bool(additional) if isinstance(additional, bool) else False,
    )

    return ToolInfo(
        name=mcp_tool.name,
        description=(getattr(mcp_tool, 'description', None) or mcp_tool.name),
        parameters=params,
    )


def _make_handler(server: MCPServer, tool_name: str):
    """Build a :class:`ToolHandler` that forwards a ``ToolCall`` into ``server.call_tool``.

    The closure captures ``server`` so the handler stays valid as long as
    the caller keeps the server entered (which the agent loop runner
    guarantees via :class:`AsyncExitStack`).
    """

    async def _handler(call: ToolCall, env: Optional[Any]) -> str:
        del env  # MCP tools never see the local sandbox environment
        return await server.call_tool(tool_name, call.function.arguments)

    return _handler


async def mcp_tools(server: MCPServer) -> Tuple[Dict[str, Any], List[ToolInfo]]:
    """Return ``(handlers_map, tool_infos)`` for all (or filtered) MCP tools.

    The returned ``handlers_map`` is keyed by the MCP tool name and maps to
    an async :class:`ToolHandler`.  The ``tool_infos`` list is the JSON-
    schema description that the loop advertises to the model.
    """
    raw_tools = await server.list_tools()
    handlers: Dict[str, Any] = {}
    infos: List[ToolInfo] = []
    for t in raw_tools:
        if not _tool_matches_filter(t.name, server.config.tools):
            continue
        handlers[t.name] = _make_handler(server, t.name)
        infos.append(_to_tool_info(t))
    return handlers, infos


async def resolve_mcp_tools(
    mcp_configs: List[MCPServerConfig],
    stack: AsyncExitStack,
) -> Tuple[Dict[str, Any], List[ToolInfo]]:
    """Spawn the configured MCP servers for one sample and return their tools.

    Each server is entered into ``stack`` so that enter / exit happen on the
    same anyio task (the sample's loop coroutine). This is what the
    underlying mcp transports (``stdio_client`` / ``streamable_http_client``
    / ``sse_client``) require — they wrap an ``anyio.create_task_group``
    whose cancel scope refuses to be exited from a different task.

    Lifetime is per-sample: every sample re-spawns its MCP servers. For
    stdio servers that costs ~0.5-1s of startup; HTTP / SSE transports only
    rebuild an httpx connection (millisecond-level). If that startup cost
    matters, point ``mcp_servers`` at a long-running remote endpoint
    (HTTP / SSE) instead of an on-demand stdio subprocess.
    """
    merged_handlers: Dict[str, Any] = {}
    merged_tool_infos: List[ToolInfo] = []

    for cfg in mcp_configs:
        server = MCPServer(cfg)
        await stack.enter_async_context(server)
        handlers, infos = await mcp_tools(server)

        for tool_name, handler in handlers.items():
            if tool_name in merged_handlers:
                logger.warning(
                    f'MCPServer[{server.name}]: tool {tool_name!r} shadows existing handler; last-write-wins'
                )
            merged_handlers[tool_name] = handler
        merged_tool_infos.extend(infos)

    return merged_handlers, merged_tool_infos


__all__ = ['mcp_tools', 'resolve_mcp_tools']
