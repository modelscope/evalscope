"""Local subprocess environment for development and testing.

Runs commands on the host OS via ``asyncio.create_subprocess_exec``.
No container isolation - suitable only for development and CI tests.
"""

import asyncio
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.agent.types import ExecResult
from evalscope.api.registry import register_environment
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_environment('local')
class LocalAgentEnvironment(AgentEnvironment):
    """Per-sample local subprocess environment.

    Executes commands as direct subprocesses on the host OS.
    Intended for **development and testing only** - no filesystem isolation.
    """

    name: str = 'local'

    def __init__(
        self,
        *,
        working_dir: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **_: Any,
    ) -> None:
        self._working_dir = working_dir
        self._env_vars: Dict[str, str] = env_vars or {}

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        effective_cwd = cwd or self._working_dir
        if self._env_vars or env:
            merged_env = {**os.environ, **self._env_vars, **(env or {})}
        else:
            merged_env = None

        loop = asyncio.get_running_loop()
        started = loop.time()
        input_bytes = input.encode('utf-8') if input is not None else None

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            # Without an explicit stdin the child inherits the evaluator's own,
            # so a model-generated command that reads stdin blocks on the
            # operator's terminal (and can swallow their keystrokes) until the
            # tool timeout fires. Nothing in a sandbox should be able to read
            # from there; DEVNULL makes such a command see EOF and fail fast.
            stdin=asyncio.subprocess.PIPE if input is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=effective_cwd,
            env=merged_env,
            start_new_session=os.name == 'posix',
        )
        communication = asyncio.create_task(proc.communicate(input_bytes))
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                asyncio.shield(communication),
                timeout=timeout,
            )
            duration = loop.time() - started
            return ExecResult(
                returncode=proc.returncode or 0,
                stdout=stdout_bytes.decode('utf-8', errors='replace'),
                stderr=stderr_bytes.decode('utf-8', errors='replace'),
                duration=duration,
            )
        except asyncio.TimeoutError:
            await self._stop_process(proc, communication)
            return ExecResult(returncode=-1, timed_out=True)
        except BaseException:
            await self._stop_process(proc, communication)
            raise

    @staticmethod
    async def _stop_process(proc: asyncio.subprocess.Process, communication: asyncio.Task[tuple[bytes, bytes]]) -> None:
        def stop(force: bool = False) -> None:
            try:
                if os.name == 'posix':
                    # The shell may have exited while descendants still hold its pipes open.
                    os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
                elif proc.returncode is None:
                    proc.kill() if force else proc.terminate()
            except ProcessLookupError:
                pass

        async def wait_for_exit() -> None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning(f'Timed out waiting for local command process {proc.pid} to exit after cleanup')

        async def cleanup() -> None:
            if os.name != 'posix':
                # Preserve direct-process cleanup where process groups are unavailable.
                # Descendants can retain the pipes, so do not wait for their EOF.
                communication.cancel()
                await asyncio.gather(communication, return_exceptions=True)
                stop()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    stop(force=True)
                finally:
                    proc._transport.close()
                if proc.returncode is None:
                    await wait_for_exit()
                return

            stop()
            try:
                # Keep draining output: waiting only for the parent can deadlock on full pipes.
                await asyncio.wait_for(asyncio.shield(communication), timeout=3.0)
            except (asyncio.TimeoutError, OSError):
                pass
            finally:
                # Also stop descendants that closed their pipes but ignored SIGTERM.
                stop(force=True)
                # communicate() does not close pipe transports when cancelled.
                proc._transport.close()
                try:
                    await asyncio.wait_for(asyncio.shield(communication), timeout=1.0)
                except (asyncio.TimeoutError, OSError):
                    communication.cancel()
                    await asyncio.gather(communication, return_exceptions=True)
                finally:
                    await wait_for_exit()

        cleanup_task = asyncio.create_task(cleanup())
        cancelled = False
        while True:
            try:
                await asyncio.shield(cleanup_task)
                break
            except asyncio.CancelledError:
                if cleanup_task.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def close(self) -> None:
        """No external resources to release."""

    async def put_dir(self, source_dir: str | Path, target_dir: str) -> None:
        """Copy a host directory into a local target directory."""
        source = Path(source_dir).expanduser()
        if not source.is_dir():
            raise FileNotFoundError(f'put_dir source is not a directory: {source}')
        target = Path(target_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)


class TemporaryLocalAgentEnvironment(LocalAgentEnvironment):
    """Local environment backed by a temporary working directory."""

    def __init__(
        self,
        sample_id: Any = None,
        *,
        prefix: str = 'evalscope-local-',
        env_vars: Optional[Dict[str, str]] = None,
    ) -> None:
        raw_sample_id = 'sample' if sample_id is None else str(sample_id)
        safe_id = ''.join(char if char.isalnum() else '-' for char in raw_sample_id)[:64]
        self._temporary_directory = tempfile.TemporaryDirectory(prefix=f'{prefix}{safe_id}-')
        super().__init__(working_dir=self._temporary_directory.name, env_vars=env_vars)

    @property
    def working_dir(self) -> Path:
        return Path(self._temporary_directory.name)

    async def close(self) -> None:
        await super().close()
        self._temporary_directory.cleanup()


__all__ = ['LocalAgentEnvironment', 'TemporaryLocalAgentEnvironment']
