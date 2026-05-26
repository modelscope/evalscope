"""Unit tests for the MCP integration.

Spawns a tiny FastMCP echo server as a stdio child process and verifies
that :class:`MCPServer` + :func:`mcp_tools` correctly:

1. Initialise the server, list tools, surface them as :class:`ToolInfo`.
2. Convert the MCP input schema into evalscope's ``ToolParams`` properties.
3. Invoke a tool and return its text content as a ``ToolHandler``-shaped
   coroutine.

A second test exercises the streamable-HTTP transport against the same
echo server bound to ``127.0.0.1`` — guards the ``client.py`` HTTP path
that the stdio test does not touch.
"""

import asyncio
import socket
import subprocess
import sys
import time
import unittest

try:
    import mcp  # noqa: F401
except ImportError:
    raise unittest.SkipTest(
        'mcp extra not installed; install with `pip install evalscope[mcp]` to run these tests.'
    )

import evalscope  # noqa: F401 - trigger registration
from evalscope.api.agent.mcp import MCPServer, MCPServerConfigHTTP, MCPServerConfigStdio, mcp_tools
from evalscope.api.tool import ToolCall
from evalscope.api.tool.tool_call import ToolFunction


def _echo_server_config() -> MCPServerConfigStdio:
    """Stdio config that spawns the local echo server fixture."""
    return MCPServerConfigStdio(
        command=sys.executable,
        args=['-m', 'tests.agent.mcp_echo_server'],
    )


def _free_port() -> int:
    """Reserve a free localhost port for the HTTP fixture."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    """Block until ``127.0.0.1:port`` accepts connections, or raise."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.settimeout(0.5)
                s.connect(('127.0.0.1', port))
                return
            except OSError:
                time.sleep(0.1)
    raise RuntimeError(f'echo HTTP server did not start on port {port} within {timeout}s')


class TestMCPServer(unittest.TestCase):
    """End-to-end MCP server lifecycle."""

    def test_list_and_call_echo_tool(self):
        async def _run():
            async with MCPServer(_echo_server_config()) as server:
                handlers, infos = await mcp_tools(server)
                # 1. tools/list — server advertises a single 'echo' tool
                self.assertIn('echo', handlers)
                self.assertEqual(len(infos), 1)
                info = infos[0]
                self.assertEqual(info.name, 'echo')
                self.assertIn('message', info.parameters.properties)
                self.assertIn('message', info.parameters.required)

                # 2. tools/call — handler returns the prefixed string
                handler = handlers['echo']
                call = ToolCall(id='c0', function=ToolFunction(name='echo', arguments={'message': 'hi'}))
                observation = await handler(call, None)
                self.assertIn('echoed: hi', observation)

        asyncio.run(_run())

    def test_filter_tools(self):
        """``tools=['echo']`` keeps the tool; an empty filter removes it."""

        async def _run():
            cfg = _echo_server_config()
            cfg.tools = ['echo']
            async with MCPServer(cfg) as server:
                handlers, _ = await mcp_tools(server)
                self.assertIn('echo', handlers)

            cfg.tools = ['nonexistent']
            async with MCPServer(cfg) as server:
                handlers, _ = await mcp_tools(server)
                self.assertNotIn('echo', handlers)

        asyncio.run(_run())

    def test_http_transport(self):
        """Streamable-HTTP transport: spawn echo server bound to a port, call ``echo``.

        Regression test for the ``client.py`` HTTP path — older versions of
        the mcp SDK accepted ``headers`` / ``timeout`` directly on
        ``streamable_http_client``; the current SDK requires those to go via
        a caller-provided ``httpx.AsyncClient``. The stdio tests do not
        exercise that code path.
        """
        port = _free_port()
        proc = subprocess.Popen(
            [sys.executable, '-m', 'tests.agent.mcp_echo_server', 'http', str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_port(port)

            cfg = MCPServerConfigHTTP(
                url=f'http://127.0.0.1:{port}/mcp',
                headers={'X-Test': 'evalscope'},
                timeout=10.0,
            )

            async def _run():
                async with MCPServer(cfg) as server:
                    handlers, infos = await mcp_tools(server)
                    self.assertIn('echo', handlers)
                    self.assertEqual(infos[0].name, 'echo')

                    call = ToolCall(
                        id='c0',
                        function=ToolFunction(name='echo', arguments={'message': 'via http'}),
                    )
                    observation = await handlers['echo'](call, None)
                    self.assertIn('echoed: via http', observation)

            asyncio.run(_run())
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == '__main__':
    unittest.main()
