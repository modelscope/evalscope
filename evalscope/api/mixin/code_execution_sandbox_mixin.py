"""Code execution sandbox mixin backed by :class:`SandboxService`.

This mixin is for benchmarks that execute generated code during scoring
(``match_score`` / verifier logic). It owns a benchmark-level sandbox pool and
does not model a full per-sample agent environment.

The mixin only:

* decides whether a benchmark wants code execution sandboxing (``use_sandbox``);
* resolves the engine + sandbox config from ``TaskConfig`` + ``BenchmarkMeta``;
* exposes a simple ``execute_code_in_sandbox`` facade for adapters.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from evalscope.api.sandbox import (
    DockerImageSpec,
    PoolHandle,
    SandboxEngine,
    build_and_acquire_pool_sync,
    merge_sandbox_config_dicts,
    prepare_docker_image,
    resolve_engine,
)
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner
from evalscope.utils.function_utils import thread_safe
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from ms_enclave.sandbox.manager import SandboxManager

    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.config import TaskConfig

logger = get_logger()


class CodeExecutionBackend(ABC):
    """Abstract base class for code execution sandbox backends.

    Kept so alternative non-ms_enclave backends can be plugged in later.
    Today there is a single implementation (:class:`EnclaveCodeExecutionBackend`).
    """

    def __init__(self, benchmark_meta: 'BenchmarkMeta', task_config: 'TaskConfig'):
        self._benchmark_meta = benchmark_meta
        self._task_config = task_config

    @abstractmethod
    def start(self) -> None:
        """Initialise and warm the sandbox backend."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True after :meth:`start` completed successfully."""

    @abstractmethod
    def execute(self, code: Union[str, List[str]], timeout: int, language: str) -> Dict[str, Any]:
        """Execute code and return a status dict."""

    @abstractmethod
    def stop(self) -> None:
        """Release any sandbox-local resources (not the service itself)."""


class EnclaveCodeExecutionBackend(CodeExecutionBackend):
    """ms_enclave-backed code execution pool delegating to :class:`SandboxService`."""

    def __init__(
        self,
        benchmark_meta: 'BenchmarkMeta',
        task_config: 'TaskConfig',
        image_spec_provider: Optional[Callable[[], Optional[DockerImageSpec]]] = None,
    ):
        super().__init__(benchmark_meta, task_config)
        self._pool_handle: Optional[PoolHandle] = None
        self._pool_size: int = self._resolve_pool_size()
        self._image_spec_provider = image_spec_provider

    def _resolve_pool_size(self) -> int:
        if not self._task_config:
            return 1
        sandbox = self._task_config.sandbox
        if sandbox is not None and sandbox.pool_size:
            return int(sandbox.pool_size)
        return int(getattr(self._task_config, 'eval_batch_size', 1) or 1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        engine = self._resolve_engine()
        sandbox_cfg_dict = self._resolve_sandbox_config_dict()
        manager_config = self._resolve_manager_config()

        if engine is SandboxEngine.DOCKER:
            image_spec = self._image_spec_provider() if self._image_spec_provider is not None else None
            if image_spec is not None:
                result = prepare_docker_image(image_spec)
                sandbox_cfg_dict = dict(sandbox_cfg_dict)
                sandbox_cfg_dict['image'] = result.image_tag
                logger.info(f'Sandbox image prepared: {result.image_tag} (reused={result.reused})')

        self._pool_handle = build_and_acquire_pool_sync(
            engine=engine,
            pool_size=self._pool_size,
            sandbox_config_dict=sandbox_cfg_dict,
            manager_config=manager_config,
        )

    def is_ready(self) -> bool:
        return self._pool_handle is not None

    def execute(self, code: Union[str, List[str]], timeout: int, language: str) -> Dict[str, Any]:
        import asyncio
        import concurrent.futures as cf
        from ms_enclave.sandbox.model import ExecutionStatus

        async def _execute_async():
            if language.lower() == 'python':
                tool_name = 'python_executor'
                parameters = {'code': code, 'timeout': timeout}
            elif language.lower() == 'shell':
                tool_name = 'shell_executor'
                parameters = {'command': code, 'timeout': timeout}
            else:
                tool_name = 'multi_code_executor'
                parameters = {'code': code, 'language': language, 'run_timeout': timeout}

            if not self._pool_handle:
                raise RuntimeError('Sandbox backend is not initialized')

            return await self._pool_handle.execute_tool(tool_name, parameters)

        try:
            result = AsyncioLoopRunner.run(_execute_async(), timeout=timeout + 10)
            return result.model_dump(exclude_none=True)
        except (TimeoutError, asyncio.TimeoutError, cf.TimeoutError) as exc:
            logger.error(f'Code execution in sandbox timed out: {exc!r}')
            return {
                'status': ExecutionStatus.TIMEOUT,
                'error': 'Code execution timed out.',
                'metadata': {
                    'code': code,
                    'language': language
                },
            }
        except Exception as exc:
            logger.exception(f'Code execution in sandbox failed: {exc!r}')
            return {
                'status': ExecutionStatus.ERROR,
                'error': str(exc),
                'metadata': {
                    'code': code,
                    'language': language
                },
            }

    def stop(self) -> None:
        # Manager lifecycle is owned by SandboxService; we only drop our pool reference.
        self._pool_handle = None

    # ------------------------------------------------------------------
    # Config resolution helpers
    # ------------------------------------------------------------------

    def _resolve_engine(self) -> SandboxEngine:
        return resolve_engine(self._task_config.sandbox.engine)

    def _resolve_manager_config(self) -> Dict[str, Any]:
        return dict(self._task_config.sandbox.manager_config or {})

    def _resolve_sandbox_config_dict(self) -> Dict[str, Any]:
        task_default = dict(self._task_config.sandbox.default_config or {})
        meta_default = dict(getattr(self._benchmark_meta, 'sandbox_config', None) or {})
        return merge_sandbox_config_dicts(meta_default, task_default)

    # ------------------------------------------------------------------
    # Accessor used by ``CodeExecutionSandboxMixin.sandbox_manager``
    # ------------------------------------------------------------------

    @property
    def manager(self) -> Optional['SandboxManager']:
        return self._pool_handle.manager if self._pool_handle else None


class CodeExecutionSandboxMixin:
    """Mixin for benchmarks that execute generated code in a sandbox pool."""

    def __init__(self, benchmark_meta: 'BenchmarkMeta', task_config: Optional['TaskConfig'] = None):
        self._benchmark_meta = benchmark_meta
        self._task_config = task_config

        self._backend: Optional[CodeExecutionBackend] = None
        """Code execution sandbox backend instance."""

        super().__init__(benchmark_meta=benchmark_meta, task_config=task_config)

    @property
    def use_sandbox(self) -> bool:
        """Return whether to use sandboxed code execution for the benchmark."""
        if not self._task_config or self._task_config.sandbox is None:
            return False
        return bool(self._task_config.sandbox.enabled)

    @property
    def sandbox_manager(self) -> Optional['SandboxManager']:
        """Get the underlying SandboxManager instance (or ``None`` if not started)."""
        if isinstance(self._backend, EnclaveCodeExecutionBackend):
            return self._backend.manager
        return None

    def _get_backend(self) -> CodeExecutionBackend:
        if self._backend:
            return self._backend
        self._backend = EnclaveCodeExecutionBackend(
            self._benchmark_meta,
            self._task_config,
            image_spec_provider=self.get_sandbox_image_spec,
        )
        return self._backend

    def get_sandbox_image_spec(self) -> Optional[DockerImageSpec]:
        """Return a benchmark-level Docker image spec for the code execution pool, if needed."""
        return None

    @thread_safe
    def ensure_sandbox_ready(self) -> bool:
        """Ensure the sandbox backend is initialized.  Thread-safe and idempotent."""
        if not self.use_sandbox:
            return False

        backend = self._get_backend()
        if backend.is_ready():
            return True

        backend.start()
        return True

    def execute_code_in_sandbox(
        self,
        code: Union[str, List[str]],
        timeout: int = 60,
        language: str = 'python',
    ) -> Dict[str, Any]:
        """Execute code in the sandbox."""
        if not self.ensure_sandbox_ready():
            logger.warning('Sandbox is not initialized.')
            return {'error': 'Sandbox is not initialized.'}

        return self._get_backend().execute(code, timeout, language)

    def sandbox_finalize(self, *args, **kwargs):
        """Finalize the sandbox backend.

        Manager lifecycle is owned by :class:`SandboxService` and cleaned up
        in an ``atexit`` hook; this only drops backend-local references.
        """
        if self._backend:
            self._backend.stop()
