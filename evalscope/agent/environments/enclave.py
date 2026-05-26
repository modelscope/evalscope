"""Unified per-sample sandbox environment backed by ms_enclave.

``EnclaveAgentEnvironment`` is the single implementation powering the
``enclave`` / ``docker`` / ``volcengine`` registry aliases, supporting any
engine understood by :class:`evalscope.api.sandbox.SandboxEngine`.

Per-sample lifecycle:

1. First :meth:`exec` call asks :class:`SandboxService` for a
   :class:`SandboxHandle` (``manager.create_sandbox``).
2. The sandbox persists for the duration of the sample.
3. :meth:`close` (called by the ``AgentAdapter`` ``finally`` block) releases
   the sandbox via ``SandboxHandle.close()``.

The underlying ``SandboxManager`` is **not** owned by the environment – it
lives in the process-wide :class:`SandboxService` and is stopped by the
shared ``atexit`` hook.
"""

from __future__ import annotations

import shlex
import time
from typing import Any, Dict, List, Optional, Union

from evalscope.api.agent import AgentEnvironment
from evalscope.api.agent.types import ExecResult
from evalscope.api.registry import register_environment
from evalscope.api.sandbox import (
    SandboxEngine,
    SandboxHandle,
    build_sandbox_config,
    get_sandbox_service,
    merge_sandbox_config_dicts,
    resolve_engine,
)
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

_DEFAULT_DOCKER_IMAGE = 'python:3.11-slim'
_DEFAULT_WORKDIR = '/workspace'
_DEFAULT_TOOLS: List[str] = ['shell_executor', 'python_executor']


@register_environment(['enclave', 'docker', 'volcengine'])
class EnclaveAgentEnvironment(AgentEnvironment):
    """Per-sample sandbox via :class:`SandboxService`.

    Parameters
    ----------
    engine:
        ``'docker'`` | ``'volcengine'`` (or any alias accepted by
        :func:`resolve_engine`).  Defaults to ``'docker'``.
    sandbox_config:
        Raw dict used to build the engine-specific ``SandboxConfig``.  For
        ``docker`` this corresponds to the fields of
        ``ms_enclave.sandbox.model.DockerSandboxConfig``; for ``volcengine``
        to ``VolcengineSandboxConfig``.
    manager_config:
        Raw dict passed through to the ms_enclave manager constructor
        (e.g. ``base_url`` / ``region`` / credentials).
    timeout:
        Default command timeout (seconds) used when :meth:`exec` is called
        without an explicit timeout.
    """

    name: str = 'enclave'

    def __init__(
        self,
        *,
        engine: Union[str, SandboxEngine] = SandboxEngine.DOCKER,
        sandbox_config: Optional[Dict[str, Any]] = None,
        manager_config: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
        **_: Any,
    ) -> None:
        # ms_enclave is mandatory for this environment. Fail fast at construction
        # time so missing-dependency errors don't surface as opaque tool errors
        # inside the agent loop (which the model interprets as a broken sandbox).
        check_import('ms_enclave', extra='sandbox', raise_error=True, feature_name='EnclaveAgentEnvironment')

        self._engine: SandboxEngine = resolve_engine(engine)
        self._timeout = float(timeout)
        self._manager_config: Dict[str, Any] = dict(manager_config or {})
        self._sandbox_config_dict: Dict[str, Any] = self._apply_engine_defaults(dict(sandbox_config or {}))
        self._handle: Optional[SandboxHandle] = None

    # ------------------------------------------------------------------
    # Config defaults
    # ------------------------------------------------------------------

    def _apply_engine_defaults(self, sandbox_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in engine-specific defaults for fields the caller didn't set.

        For ``docker`` we inject ``image`` / ``working_dir`` / ``tools_config``
        so that a bare ``EnclaveAgentEnvironment(engine='docker')`` produces a
        usable sandbox out of the box.  User-provided values always win.
        """
        if self._engine is SandboxEngine.DOCKER:
            defaults: Dict[str, Any] = {
                'image': _DEFAULT_DOCKER_IMAGE,
                'working_dir': _DEFAULT_WORKDIR,
                'tools_config': list(_DEFAULT_TOOLS),
            }
            return merge_sandbox_config_dicts(defaults, sandbox_config)
        return sandbox_config

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    def merge_sandbox_config(self, overlay: Dict[str, Any]) -> None:
        """Merge ``overlay`` into the pending sandbox config.

        Used by the adapter layer to inject env-specific fields (e.g.
        ``extra_hosts`` for ``host.docker.internal:host-gateway`` on
        Linux) **before** the sandbox is created. Raises if the sandbox
        is already running — caller would otherwise see no effect.
        """
        if self._handle is not None:
            raise RuntimeError(
                'EnclaveAgentEnvironment.merge_sandbox_config: sandbox '
                f'{self._handle.sandbox_id} is already running; merge before exec.'
            )
        self._sandbox_config_dict = merge_sandbox_config_dicts(self._sandbox_config_dict, overlay)

    async def _ensure_sandbox(self) -> SandboxHandle:
        if self._handle is None:
            service = get_sandbox_service()
            sandbox_config = build_sandbox_config(self._engine, self._sandbox_config_dict)
            self._handle = await service.create_sandbox(
                engine=self._engine,
                sandbox_config=sandbox_config,
                manager_config=self._manager_config or None,
            )
            logger.debug(
                f'EnclaveAgentEnvironment: sandbox {self._handle.sandbox_id} ready '
                f'(engine={self._engine.value}).'
            )
        return self._handle

    # ------------------------------------------------------------------
    # AgentEnvironment interface
    # ------------------------------------------------------------------

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,  # noqa: A002 - mirrors ABC signature
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        handle = await self._ensure_sandbox()

        if isinstance(cmd, list):
            command = ' '.join(shlex.quote(c) for c in cmd)
        else:
            command = str(cmd)
        if env:
            # Prepend ``KEY=VAL`` pairs so the agent CLI inherits them.
            # Using ``env -- ... <cmd>`` would be cleaner but is not always
            # available inside minimal images; inline assignment works in any
            # POSIX shell.
            prefix = ' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())
            command = f'{prefix} {command}' if prefix else command
        if cwd:
            command = f'cd {shlex.quote(cwd)} && {command}'

        # ms_enclave's shell_executor splits a bare string with no shell
        # wrapping; wrap as ``bash -c`` so cd/&&/env-prefix/quoting survive.
        shell_argv = ['bash', '-c', command]

        timeout_s = float(timeout or self._timeout)

        from ms_enclave.sandbox.model import ExecutionStatus

        # ms_enclave's ExecutionResult.execution_time is unreliable (often
        # ``None`` / ``0`` for shell_executor). Wall-time it locally so
        # ``ExecResult.duration`` always reflects real elapsed time.
        started = time.monotonic()
        result = await handle.execute_tool(
            'shell_executor',
            {
                'command': shell_argv,
                'timeout': timeout_s,
            },
        )
        elapsed = time.monotonic() - started

        stdout = str(result.output or '')
        stderr = str(result.error or '')
        timed_out = result.status == ExecutionStatus.TIMEOUT

        if result.status == ExecutionStatus.SUCCESS:
            returncode = 0
        elif timed_out:
            returncode = -1
        else:
            import re as _re
            _m = _re.search(r'exit code (\d+)', stderr)
            returncode = int(_m.group(1)) if _m else 1

        # Prefer upstream-reported time when it's a positive number;
        # otherwise fall back to our locally measured wall-clock.
        upstream_duration = float(result.execution_time or 0.0)
        duration = upstream_duration if upstream_duration > 0 else elapsed

        return ExecResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration=duration,
        )

    async def close(self) -> None:
        """Release the per-sample sandbox (idempotent)."""
        if self._handle is not None:
            try:
                await self._handle.close()
            finally:
                self._handle = None


__all__ = ['EnclaveAgentEnvironment']
