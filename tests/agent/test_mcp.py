"""Unit tests for the MCP integration.

Spawns a tiny FastMCP echo server as a stdio child process and verifies
that :class:`MCPServer` + :func:`mcp_tools` correctly:

1. Initialise the server, list tools, surface them as :class:`ToolInfo`.
2. Convert the MCP input schema into evalscope's ``ToolParams`` properties.
3. Invoke a tool and return its text content as a ``ToolHandler``-shaped
   coroutine.
"""

import asyncio
import sys
import unittest

import evalscope  # noqa: F401 - trigger registration

from evalscope.api.agent.mcp import MCPServer, MCPServerConfigStdio, mcp_tools
from evalscope.api.tool import ToolCall
from evalscope.api.tool.tool_call import ToolFunction


def _echo_server_config() -> MCPServerConfigStdio:
    """Stdio config that spawns the local echo server fixture."""
    return MCPServerConfigStdio(
        command=sys.executable,
        args=['-m', 'tests.agent.mcp_echo_server'],
    )


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


if __name__ == '__main__':
    unittest.main()
