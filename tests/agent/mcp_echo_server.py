"""Tiny FastMCP server used as a stdio fixture by ``test_mcp.py``.

Exposes a single ``echo`` tool that returns the input ``message`` prefixed
by ``echoed: ``. Run as ``python -m tests.agent.mcp_echo_server`` and
talk over stdio.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP('evalscope-test-echo')


@mcp.tool()
def echo(message: str) -> str:
    """Return ``message`` with an ``echoed:`` prefix."""
    return f'echoed: {message}'


if __name__ == '__main__':
    mcp.run()
