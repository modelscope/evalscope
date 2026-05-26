"""Pydantic configs for MCP servers attached via ``NativeAgentConfig.mcp_servers``."""

from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Dict, List, Literal, Optional, Union


class _MCPServerConfigBase(BaseModel):
    """Fields shared by every MCP server config variant."""

    model_config = ConfigDict(extra='forbid')

    name: Optional[str] = Field(default=None)
    """Human-readable display name (used in logs / traces). Defaults to ``command`` or ``url``."""

    tools: Union[Literal['all'], List[str]] = Field(default='all')
    """Whitelist of tool names exposed to the model. ``'all'`` exports every tool."""


class MCPServerConfigStdio(_MCPServerConfigBase):
    """Spawn a local MCP server as a child process and talk over stdio.

    Mirrors :class:`mcp.client.stdio.StdioServerParameters`.
    """

    type: Literal['stdio'] = Field(default='stdio')

    command: str
    """Executable to spawn (e.g. ``npx`` / ``uvx`` / absolute path)."""

    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = Field(default=None)


class MCPServerConfigHTTP(_MCPServerConfigBase):
    """Connect to a remote MCP server over Streamable HTTP."""

    type: Literal['http'] = Field(default='http')

    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: float = Field(default=30.0)
    """HTTP read timeout in seconds."""


class MCPServerConfigSSE(_MCPServerConfigBase):
    """Connect to a remote MCP server over Server-Sent Events (SSE).

    SSE is the legacy MCP HTTP transport; prefer :class:`MCPServerConfigHTTP`
    (Streamable HTTP) when the server supports both.
    """

    type: Literal['sse'] = Field(default='sse')

    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: float = Field(default=5.0)
    """HTTP timeout in seconds for non-streaming operations."""
    sse_read_timeout: float = Field(default=300.0)
    """How long (seconds) to wait for the next SSE event before disconnecting."""


MCPServerConfig = Annotated[
    Union[MCPServerConfigStdio, MCPServerConfigHTTP, MCPServerConfigSSE],
    Field(discriminator='type'),
]

__all__ = [
    'MCPServerConfig',
    'MCPServerConfigStdio',
    'MCPServerConfigHTTP',
    'MCPServerConfigSSE',
]
