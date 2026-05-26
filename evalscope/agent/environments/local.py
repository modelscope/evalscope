"""Local (subprocess) environment for development and testing.

Runs commands on the host OS via ``asyncio.create_subprocess_exec``.
No container isolation - suitable only for development and CI tests.
"""

import asyncio
import os
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

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if input is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=effective_cwd,
            env=merged_env,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input.encode('utf-8') if input is not None else None),
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
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            return ExecResult(returncode=-1, timed_out=True)

    async def close(self) -> None:
        """No external resources to release."""


__all__ = ['LocalAgentEnvironment']
