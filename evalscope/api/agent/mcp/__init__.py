"""MCP (Model Context Protocol) integration for the AgentLoop.

Lets users configure :class:`MCPServerConfigStdio` / :class:`MCPServerConfigHTTP`
on ``NativeAgentConfig.mcp_servers`` and have those servers' tools
auto-merged into every ``AgentLoopAdapter``-based benchmark — without any
benchmark-side code change.

Soft dependency: the official ``mcp`` Python SDK (``pip install mcp`` or
``pip install evalscope[mcp]``) is imported lazily inside :class:`MCPServer`,
so users who don't configure any MCP server pay no cost.
"""

from .client import MCPServer
from .source import mcp_tools, resolve_mcp_tools
from .types import MCPServerConfig, MCPServerConfigHTTP, MCPServerConfigSSE, MCPServerConfigStdio

__all__ = [
    'MCPServer',
    'mcp_tools',
    'resolve_mcp_tools',
    'MCPServerConfig',
    'MCPServerConfigStdio',
    'MCPServerConfigHTTP',
    'MCPServerConfigSSE',
]
