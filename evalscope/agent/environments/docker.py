"""Docker sandbox environment backed by ms_enclave.

Each ``DockerAgentEnvironment`` instance represents **one** per-sample
container.  A class-level ``SandboxManager`` is shared across all instances
so container management overhead is amortised.

Lifecycle:
    1. First ``exec`` / ``read_file`` / ``write_file`` call creates the
       container via ``manager.create_sandbox()``.
    2. The container persists for the duration of the sample.
    3. ``close()`` (called by :class:`AgentAdapter` ``finally`` block) calls
       ``manager.delete_sandbox()`` to destroy it.
"""

import asyncio
import base64
import shlex
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.agent.types import ExecResult
from evalscope.api.registry import register_environment
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from ms_enclave.sandbox.manager import SandboxManager

logger = get_logger()

_DEFAULT_IMAGE = 'python:3.11-slim'
_DEFAULT_WORKDIR = '/workspace'


@register_environment('docker')
class DockerAgentEnvironment(AgentEnvironment):
    """Per-sample Docker sandbox via ms_enclave.

    A single :class:`~ms_enclave.sandbox.manager.SandboxManager` is shared at
    the class level (one per process).  Each instance allocates its own
    container on first use and releases it in :meth:`close`.
    """

    name: str = 'docker'

    # -----------------------------------------------------------------------
    # Class-level shared manager (one per process)
    # -----------------------------------------------------------------------
    _manager: ClassVar[Optional['SandboxManager']] = None
    _manager_lock: ClassVar[Optional[asyncio.Lock]] = None

    #: Default tools enabled in each per-sample container.
    #: ``DockerSandboxConfig.tools_config`` accepts a list of tool names;
    #: they are initialised by ``initialize_tools()`` inside ms_enclave.
    DEFAULT_TOOLS: List[str] = ['shell_executor', 'python_executor']

    def __init__(
        self,
        *,
        image: str = _DEFAULT_IMAGE,
        working_dir: str = _DEFAULT_WORKDIR,
        timeout: float = 60.0,
        env_vars: Optional[Dict[str, str]] = None,
        manager_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
        **_: Any,
    ) -> None:
        self._image = image
        self._working_dir = working_dir
        self._timeout = timeout
        self._env_vars: Dict[str, str] = env_vars or {}
        self._manager_config: Dict[str, Any] = manager_config or {}
        self._tools_config: List[str] = tools if tools is not None else list(self.DEFAULT_TOOLS)
        self._sandbox_id: Optional[str] = None

    # -----------------------------------------------------------------------
    # Class-level manager management
    # -----------------------------------------------------------------------

    @classmethod
    async def _ensure_manager(cls, manager_config: Dict[str, Any]) -> 'SandboxManager':
        """Return the shared manager, initialising it once if needed.

        Thread-safe via an :class:`asyncio.Lock` (all callers run on the same
        background event loop provided by :class:`~evalscope.utils.function_utils.AsyncioLoopRunner`).
        """
        # asyncio.Lock must be created inside the running loop.
        if cls._manager_lock is None:
            cls._manager_lock = asyncio.Lock()

        async with cls._manager_lock:
            if cls._manager is None:
                from ms_enclave.sandbox.manager import SandboxManagerFactory

                m = SandboxManagerFactory.create_manager(**manager_config)
                await m.start()
                cls._manager = m
                logger.info('DockerAgentEnvironment: shared SandboxManager started.')

        return cls._manager

    @classmethod
    async def shutdown_manager(cls) -> None:
        """Gracefully stop the shared manager.

        Call once at process exit (e.g. via ``atexit`` or a pytest fixture).
        Idempotent.
        """
        if cls._manager is not None:
            try:
                await cls._manager.stop()
                logger.info('DockerAgentEnvironment: shared SandboxManager stopped.')
            except Exception as exc:
                logger.warning(f'DockerAgentEnvironment: error stopping manager: {exc}')
            finally:
                cls._manager = None
                cls._manager_lock = None

    # -----------------------------------------------------------------------
    # Per-sample sandbox lifecycle
    # -----------------------------------------------------------------------

    async def _ensure_sandbox(self) -> str:
        """Create (once) and return the per-sample sandbox ID."""
        if self._sandbox_id is None:
            from ms_enclave.sandbox.model import DockerSandboxConfig, SandboxType

            manager = await self._ensure_manager(self._manager_config)
            config = DockerSandboxConfig(
                image=self._image,
                working_dir=self._working_dir,
                env_vars=self._env_vars,
                tools_config=self._tools_config,
            )
            self._sandbox_id = await manager.create_sandbox(SandboxType.DOCKER, config)
            logger.debug(
                f'DockerAgentEnvironment: container {self._sandbox_id} created '
                f'(image={self._image!r}, workdir={self._working_dir!r})'
            )
        return self._sandbox_id

    # -----------------------------------------------------------------------
    # AgentEnvironment interface
    # -----------------------------------------------------------------------

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,  # noqa: A002 - mirrors ABC signature
        timeout: Optional[float] = None,
    ) -> ExecResult:
        sandbox_id = await self._ensure_sandbox()
        manager = await self._ensure_manager(self._manager_config)

        if isinstance(cmd, list):
            command = ' '.join(shlex.quote(c) for c in cmd)
        else:
            command = str(cmd)

        if cwd:
            command = f'cd {shlex.quote(cwd)} && {command}'

        timeout_s = float(timeout or self._timeout)

        from ms_enclave.sandbox.model import ExecutionStatus

        result = await manager.execute_tool(
            sandbox_id,
            'shell_executor',
            {
                'command': command,
                'timeout': timeout_s
            },
        )

        # ToolResult.output is the stdout string; .error is stderr / error message.
        stdout = str(result.output or '')
        stderr = str(result.error or '')
        timed_out = result.status == ExecutionStatus.TIMEOUT

        # Determine returncode.  ms_enclave normalises all non-zero exits to
        # ExecutionStatus.ERROR but embeds the actual exit code in the error
        # message as "Command failed with exit code N".  Parse it so callers
        # can branch on specific codes.
        if result.status == ExecutionStatus.SUCCESS:
            returncode = 0
        elif timed_out:
            returncode = -1
        else:
            import re as _re
            _m = _re.search(r'exit code (\d+)', stderr)
            returncode = int(_m.group(1)) if _m else 1

        return ExecResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration=float(result.execution_time or 0.0),
        )

    async def read_file(self, path: str) -> str:
        result = await self.exec(['cat', path])
        if result.returncode != 0:
            raise FileNotFoundError(f"Cannot read file '{path}': {result.stderr.strip()}")
        return result.stdout

    async def write_file(self, path: str, content: str) -> None:
        """Write a UTF-8 file using the in-container Python interpreter.

        We go through ``python_executor`` rather than a here-string so the
        content is base64-encoded and shell escaping is a non-issue.
        """
        b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
        code = (
            'import base64, os\n'
            f'p = {repr(path)}\n'
            'os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)\n'
            f'open(p, "wb").write(base64.b64decode({repr(b64)}))\n'
        )

        sandbox_id = await self._ensure_sandbox()
        manager = await self._ensure_manager(self._manager_config)

        from ms_enclave.sandbox.model import ExecutionStatus

        result = await manager.execute_tool(
            sandbox_id,
            'python_executor',
            {
                'code': code,
                'timeout': 30.0
            },
        )
        if result.status != ExecutionStatus.SUCCESS:
            raise RuntimeError(f"write_file('{path}') failed: {result.error}")

    async def close(self) -> None:
        """Delete the per-sample container (idempotent)."""
        if self._sandbox_id is not None and self._manager is not None:
            try:
                await self._manager.delete_sandbox(self._sandbox_id)
                logger.debug(f'DockerAgentEnvironment: container {self._sandbox_id} deleted.')
            except Exception as exc:
                logger.warning(f'DockerAgentEnvironment: error deleting container '
                               f'{self._sandbox_id}: {exc}')
            finally:
                self._sandbox_id = None


__all__ = ['DockerAgentEnvironment']
