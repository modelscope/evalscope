"""Async client wrapper around the official ``mcp`` Python SDK.

Provides a single ``MCPServer`` async context manager that hides the
stdio / Streamable HTTP / SSE transport plumbing and exposes
``list_tools`` / ``call_tool`` that return EvalScope-friendly types.
"""

from __future__ import annotations

import datetime
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger
from .types import MCPServerConfig, MCPServerConfigHTTP, MCPServerConfigSSE, MCPServerConfigStdio

logger = get_logger()


class MCPServer:
    """Async context manager wrapping a single MCP ``ClientSession``.

    Lifecycle::

        async with MCPServer(config) as server:
            tools = await server.list_tools()
            text = await server.call_tool('fetch', {'url': '...'})

    The ``mcp`` Python SDK (and its stdio / streamable_http transports)
    is imported lazily so that ``mcp`` becomes a soft optional dependency:
    benchmarks that don't use MCP servers do not need the package
    installed.
    """

    def __init__(self, config: MCPServerConfig, *, call_timeout: float = 60.0) -> None:
        self.config = config
        self._call_timeout = call_timeout
        self._stack: Optional[AsyncExitStack] = None
        self._session: Any = None  # mcp.ClientSession; typed as Any to keep import lazy

    @property
    def name(self) -> str:
        if self.config.name:
            return self.config.name
        if isinstance(self.config, MCPServerConfigStdio):
            return self.config.command
        return self.config.url

    async def __aenter__(self) -> 'MCPServer':
        check_import('mcp', extra='mcp', raise_error=True, feature_name='MCP server support')
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        self._stack = AsyncExitStack()
        try:
            if isinstance(self.config, MCPServerConfigStdio):
                params = StdioServerParameters(
                    command=self.config.command,
                    args=list(self.config.args),
                    env=dict(self.config.env) or None,
                    cwd=self.config.cwd,
                )
                read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
            elif isinstance(self.config, MCPServerConfigHTTP):
                import httpx
                from mcp.client.streamable_http import streamable_http_client

                # Newer mcp SDK's ``streamable_http_client`` only accepts
                # ``url`` / ``http_client`` / ``terminate_on_close``: per-call
                # ``headers`` and ``timeout`` were moved to a caller-provided
                # ``httpx.AsyncClient``. We enter the client into our stack
                # so its connection pool is closed alongside the MCP session.
                http_client = httpx.AsyncClient(
                    headers=dict(self.config.headers) or None,
                    timeout=self.config.timeout,
                )
                await self._stack.enter_async_context(http_client)

                streams_ctx = streamable_http_client(
                    url=self.config.url,
                    http_client=http_client,
                )
                read_stream, write_stream, _ = await self._stack.enter_async_context(streams_ctx)
            elif isinstance(self.config, MCPServerConfigSSE):
                from mcp.client.sse import sse_client

                streams_ctx = sse_client(
                    url=self.config.url,
                    headers=dict(self.config.headers) or None,
                    timeout=self.config.timeout,
                    sse_read_timeout=self.config.sse_read_timeout,
                )
                read_stream, write_stream = await self._stack.enter_async_context(streams_ctx)
            else:
                raise TypeError(f'Unexpected MCP server config type: {type(self.config)!r}')

            self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
            await self._session.initialize()
            logger.info(f'MCPServer[{self.name}]: initialised')
            return self
        except Exception:
            # Make sure we don't leak any process / socket on partial init failure.
            await self._stack.aclose()
            self._stack = None
            self._session = None
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            finally:
                self._stack = None
                self._session = None
                logger.debug(f'MCPServer[{self.name}]: closed')

    async def list_tools(self) -> List[Any]:
        """Return the raw ``mcp.types.Tool`` list from the server."""
        if self._session is None:
            raise RuntimeError('MCPServer not entered; use `async with MCPServer(cfg) as server:`')
        result = await self._session.list_tools()
        return list(result.tools)

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Invoke a tool and return a flat text observation.

        Concatenates all ``TextContent`` parts from the result; non-text
        content blocks are summarised by their ``type``.  When the MCP
        server signals an error (``CallToolResult.isError``), the textual
        error body is returned with an ``[error]`` prefix so the agent
        can react to it just like a normal tool failure.
        """
        if self._session is None:
            raise RuntimeError('MCPServer not entered; use `async with MCPServer(cfg) as server:`')

        result = await self._session.call_tool(
            name=name,
            arguments=arguments or {},
            read_timeout_seconds=datetime.timedelta(seconds=self._call_timeout),
        )

        text_parts: List[str] = []
        for block in result.content or []:
            block_type = getattr(block, 'type', None)
            if block_type == 'text':
                text_parts.append(getattr(block, 'text', '') or '')
            else:
                text_parts.append(f'[{block_type or "non-text"} content omitted]')

        body = '\n'.join(p for p in text_parts if p) or '(empty MCP response)'
        if result.isError:
            return f'[error] {body}'
        return body


__all__ = ['MCPServer']
