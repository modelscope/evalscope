"""Regression tests for stderr capture in service task subprocesses."""

import asyncio
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

from evalscope.service.utils.process import _capture_stderr, run_in_subprocess


def _call_stdio_echo() -> list[str]:
    # The SDK must first import inside the worker's stderr capture context.
    assert 'mcp.client.stdio' not in sys.modules
    from evalscope.api.agent.mcp import MCPServer, MCPServerConfigStdio

    async def run() -> list[str]:
        config = MCPServerConfigStdio(
            command=sys.executable,
            args=[str(Path(__file__).resolve().parents[1] / 'agent' / 'mcp_echo_server.py')],
        )
        results = []
        for message in ('first sample', 'second sample'):
            async with MCPServer(config) as server:
                tool_names = [tool.name for tool in await server.list_tools()]
                assert 'echo' in tool_names
                results.append(await server.call_tool('echo', {'message': message}))
        return results

    return asyncio.run(asyncio.wait_for(run(), timeout=30))


def _fail_with_stderr() -> None:
    print('task stderr: 诊断信息', file=sys.stderr)
    subprocess.run(
        [sys.executable, '-c', "import sys; print('nested process stderr', file=sys.stderr)"],
        stderr=sys.stderr,
        check=True,
        timeout=10,
    )
    raise ValueError('task failed')


class TestProcessStderr(unittest.TestCase):
    """Capture supports subprocess file descriptors and preserves diagnostics."""

    @unittest.skipUnless(importlib.util.find_spec('mcp'), 'mcp extra not installed')
    def test_spawned_worker_can_start_stdio_mcp(self) -> None:
        """Lazy SDK imports and repeated MCP sessions work in a service worker."""
        self.assertEqual(
            run_in_subprocess(_call_stdio_echo),
            ['echoed: first sample', 'echoed: second sample'],
        )

    def test_worker_forwards_task_and_nested_process_stderr(self) -> None:
        """Task failures retain stderr from Python and nested subprocesses."""
        with self.assertRaises(RuntimeError) as raised:
            run_in_subprocess(_fail_with_stderr)
        message = str(raised.exception)
        self.assertIn('ValueError: task failed', message)
        self.assertIn('[stderr]', message)
        self.assertIn('task stderr: 诊断信息', message)
        self.assertIn('nested process stderr', message)

    def test_capture_restores_stderr_on_exception(self) -> None:
        """An exception must not leave the process using redirected stderr."""
        original = sys.stderr
        with self.assertRaisesRegex(ValueError, 'capture failed'):
            with _capture_stderr():
                raise ValueError('capture failed')
        self.assertIs(sys.stderr, original)
