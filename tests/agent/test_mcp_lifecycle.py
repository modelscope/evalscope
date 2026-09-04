"""MCP initialization failures must preserve errors and release transport resources."""

import asyncio
import unittest
from typing import List
from unittest.mock import patch

import httpx
import pytest

pytest.importorskip('mcp')

from evalscope.api.agent.mcp import MCPServer, MCPServerConfigHTTP
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner


def _exception_leaves(error: BaseException) -> List[BaseException]:
    children = getattr(error, 'exceptions', None)
    if children is None:
        return [error]
    return [leaf for child in children for leaf in _exception_leaves(child)]


class TestMCPInitialization(unittest.TestCase):
    """Exercise the real MCP SDK with deterministic HTTP transports."""

    def tearDown(self) -> None:
        AsyncioLoopRunner.shutdown_for_thread()

    def test_http_failure_preserves_cause(self) -> None:
        """The synchronous bridge must expose connection and HTTP errors, not cancellation."""
        for status in (None, 401, 500):
            with self.subTest(status=status):
                def respond(request: httpx.Request) -> httpx.Response:
                    if status is None:
                        raise httpx.ConnectError('connection refused', request=request)
                    return httpx.Response(status, request=request)

                client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
                server = MCPServer(MCPServerConfigHTTP(url='http://mcp.test/mcp'))

                async def run() -> None:
                    async with server:
                        self.fail('Initialization must fail')

                with patch('httpx.AsyncClient', return_value=client):
                    with self.assertRaises(BaseException) as caught:
                        AsyncioLoopRunner.run(run(), timeout=5)

                expected = httpx.ConnectError if status is None else httpx.HTTPStatusError
                leaves = _exception_leaves(caught.exception)
                self.assertTrue(any(isinstance(error, expected) for error in leaves), repr(caught.exception))
                self.assertTrue(client.is_closed)
                with self.assertRaisesRegex(RuntimeError, 'MCPServer not entered'):
                    AsyncioLoopRunner.run(server.list_tools(), timeout=5)

    def test_cancel_initialization_closes_transport(self) -> None:
        """External cancellation stays cancellation, while the HTTP client is closed."""
        async def run() -> None:
            started = asyncio.Event()

            async def respond(request: httpx.Request) -> httpx.Response:
                started.set()
                await asyncio.Future()
                raise AssertionError('Request must be cancelled')

            client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            server = MCPServer(MCPServerConfigHTTP(url='http://mcp.test/mcp'))

            async def connect() -> None:
                async with server:
                    self.fail('Initialization must be cancelled')

            with patch('httpx.AsyncClient', return_value=client):
                task = asyncio.create_task(connect())
                try:
                    await asyncio.wait_for(started.wait(), timeout=5)
                finally:
                    task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertTrue(client.is_closed)
            with self.assertRaisesRegex(RuntimeError, 'MCPServer not entered'):
                await server.list_tools()

        AsyncioLoopRunner.run(run(), timeout=10)
