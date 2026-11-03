"""Local tool execution must not leave work running after timeout or cancellation."""

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from evalscope.agent.environments.local import LocalAgentEnvironment

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX process-group lifecycle')


def _running(pid: int) -> bool:
    # An orphan can briefly remain a zombie until init reaps it; it is no longer executing.
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True)
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith('Z')


async def _wait_ready(path: Path) -> list[int]:
    for _ in range(500):
        if path.exists():
            pids = path.read_text().split()
            if len(pids) == 2:
                return [int(pid) for pid in pids]
        await asyncio.sleep(0.01)
    raise AssertionError('Subprocess did not become ready')


@pytest.mark.parametrize('stop', ['timeout', 'cancel'])
@pytest.mark.parametrize('parent_exits', [False, True])
@pytest.mark.parametrize('ignore_term', [False, True])
def test_stopping_exec_reaps_process_tree(
    tmp_path: Path, stop: str, parent_exits: bool, ignore_term: bool
) -> None:
    ready = tmp_path / 'ready'
    child_ready = tmp_path / 'child-ready'
    child = (
        'import signal, time; from pathlib import Path; '
        + ('signal.signal(signal.SIGTERM, signal.SIG_IGN); ' if ignore_term else '')
        + f'Path({str(child_ready)!r}).touch(); time.sleep(60)'
    )
    parent = (
        'import os, subprocess, sys, time; from pathlib import Path; '
        f'p = subprocess.Popen([sys.executable, "-c", {child!r}]); '
        f'child_ready = Path({str(child_ready)!r}); '
        '\nwhile not child_ready.exists(): time.sleep(0.01)\n'
        f'Path({str(ready)!r}).write_text(str(os.getpid()) + " " + str(p.pid)); '
        + ('sys.exit(0)' if parent_exits else 'time.sleep(60)')
    )

    async def run() -> None:
        task = asyncio.create_task(
            LocalAgentEnvironment().exec([sys.executable, '-c', parent], timeout=1 if stop == 'timeout' else None)
        )
        pids: list[int] = []
        try:
            pids = await _wait_ready(ready)
            if stop == 'cancel':
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=6)
            else:
                result = await asyncio.wait_for(task, timeout=6)
                assert result.timed_out
                assert result.returncode == -1
            assert not any(_running(pid) for pid in pids), 'Tool processes are still running after exec ended'
        finally:
            # Also clean up the intentionally failing baseline so the regression leaves no orphans.
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0.05)

    asyncio.run(run())


def test_non_posix_cleanup_does_not_wait_for_descendant_pipes(monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.agent.environments import local

    async def run() -> None:
        async def communicate() -> tuple[bytes, bytes]:
            await asyncio.Event().wait()
            return b'', b''

        communication = asyncio.create_task(communicate())
        proc = AsyncMock(spec=asyncio.subprocess.Process)
        proc.returncode = 0
        with monkeypatch.context() as patch:
            patch.setattr(local.os, 'name', 'nt')
            await asyncio.wait_for(LocalAgentEnvironment._stop_process(proc, communication), timeout=0.5)
        assert communication.cancelled()
        proc.wait.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize('cancel_drain', [False, True])
def test_cleanup_always_reaps_parent(monkeypatch: pytest.MonkeyPatch, cancel_drain: bool) -> None:
    """Pipe errors or repeated cancellation must not bypass the final process wait."""
    from evalscope.agent.environments import local

    async def run() -> None:
        draining = asyncio.Event()
        release = asyncio.Event()

        async def communicate() -> tuple[bytes, bytes]:
            await draining.wait()
            await release.wait()
            if not cancel_drain:
                raise OSError('pipe read failed')
            return b'', b''

        communication = asyncio.create_task(communicate())
        proc = AsyncMock(spec=asyncio.subprocess.Process)
        proc.pid = 123

        def signal_group(pid: int, sig: int) -> None:
            if sig == signal.SIGKILL:
                draining.set()

        monkeypatch.setattr(local.os, 'killpg', signal_group)
        cleanup = asyncio.create_task(LocalAgentEnvironment._stop_process(proc, communication))
        await draining.wait()
        if cancel_drain:
            cleanup.cancel()
            await asyncio.sleep(0)
            cleanup.cancel()
            await asyncio.sleep(0)
        release.set()
        if cancel_drain:
            with pytest.raises(asyncio.CancelledError):
                await cleanup
        else:
            await cleanup
        assert not communication.cancelled(), 'Cancellation interrupted pipe cleanup'
        proc.wait.assert_awaited_once()

    asyncio.run(run())
