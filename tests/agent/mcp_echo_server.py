"""Tiny FastMCP server used as a fixture by ``test_mcp.py``.

Exposes a single ``echo`` tool that returns the input ``message`` prefixed
by ``echoed: ``. Runs in stdio by default; pass ``http <port>`` to bind
streamable-HTTP on ``127.0.0.1:<port>/mcp`` instead.
"""

import sys
from mcp.server.fastmcp import FastMCP


def _build_server(port: int = 8000) -> FastMCP:
    mcp = FastMCP('evalscope-test-echo', host='127.0.0.1', port=port)

    @mcp.tool()
    def echo(message: str) -> str:
        """Return ``message`` with an ``echoed:`` prefix."""
        return f'echoed: {message}'

    return mcp


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'http':
        port = int(sys.argv[2]) if len(sys.argv) >= 3 else 8000
        _build_server(port).run(transport='streamable-http')
    else:
        _build_server().run()
