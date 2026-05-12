from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from evalscope.utils.function_utils import AsyncioLoopRunner, thread_safe
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from ms_enclave.sandbox.manager import SandboxManager

    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.config import TaskConfig

logger = get_logger()


class SandboxBackend(ABC):
    """Abstract base class for sandbox backends."""

    def __init__(self, benchmark_meta: 'BenchmarkMeta', task_config: 'TaskConfig'):
        self._benchmark_meta = benchmark_meta
        self._task_config = task_config

    @abstractmethod
    def start(self) -> None:
        """Initialize and start the sandbox backend."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the backend is ready."""
        pass

    @abstractmethod
    def execute(self, code: Union[str, List[str]], timeout: int, language: str) -> Dict[str, Any]:
        """Execute code in the sandbox."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop and finalize the sandbox backend."""
        pass


class EnclaveSandboxBackend(SandboxBackend):
    """Sandbox backend using ms_enclave (Docker or Volcengine/SandboxFusion)."""

    def __init__(self, benchmark_meta: 'BenchmarkMeta', task_config: 'TaskConfig'):
        super().__init__(benchmark_meta, task_config)
        self._manager: Optional['SandboxManager'] = None
        self._pool_size: int = self._task_config.eval_batch_size if self._task_config else 1
        self._use_custom_image: bool = False

    def start(self) -> None:
        self.init_sandbox_manager()
        self.init_sandbox()

    def is_ready(self) -> bool:
        return self._manager is not None and self._manager._pool_initialized

    def execute(self, code: Union[str, List[str]], timeout: int, language: str) -> Dict[str, Any]:
        import asyncio
        import concurrent.futures as cf
        from ms_enclave.sandbox.model import ExecutionStatus, ToolResult

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

            if not self._manager:
                raise RuntimeError('Sandbox manager is not initialized')

            result = await self._manager.execute_tool_in_pool(tool_name, parameters)
            return result

        try:
            result = AsyncioLoopRunner.run(_execute_async(), timeout=timeout + 10)
            return result.model_dump(exclude_none=True)
        except (TimeoutError, asyncio.TimeoutError, cf.TimeoutError) as e:
            logger.error(f'Code execution in sandbox timed out: {e!r}')
            return {
                'status': ExecutionStatus.TIMEOUT,
                'error': 'Code execution timed out.',
                'metadata': {
                    'code': code,
                    'language': language
                }
            }
        except Exception as e:
            logger.exception(f'Code execution in sandbox failed: {e!r}')
            return {'status': ExecutionStatus.ERROR, 'error': str(e), 'metadata': {'code': code, 'language': language}}

    def stop(self) -> None:
        if self._manager:
            try:
                AsyncioLoopRunner.run(self._manager.stop(), timeout=600)
                logger.info('Sandbox manager finalized.')
            except Exception as e:
                logger.warning(f'Error finalizing sandbox manager: {e}')

    def _is_volcengine_sandbox(self) -> bool:
        sandbox_type = self._task_config.sandbox_type if self._task_config else None
        if not sandbox_type:
            return False
        return str(sandbox_type).lower() in {'volcengine', 'volcano', 'volc'}

    async def init_sandbox_manager_async(self) -> Optional['SandboxManager']:
        if self._manager is not None:
            return self._manager

        manager_config = self._task_config.sandbox_manager_config or {}

        if self._is_volcengine_sandbox():
            from ms_enclave.sandbox.manager import VolcengineSandboxManager
            from ms_enclave.sandbox.model import VolcengineSandboxManagerConfig

            config = VolcengineSandboxManagerConfig(**manager_config)
            self._manager = VolcengineSandboxManager(config=config)
        else:
            from ms_enclave.sandbox.manager import SandboxManagerFactory

            self._manager = SandboxManagerFactory.create_manager(**manager_config)

        await self._manager.start()
        logger.info('Sandbox manager initialized.')
        return self._manager

    def init_sandbox_manager(self) -> Optional['SandboxManager']:
        return AsyncioLoopRunner.run(self.init_sandbox_manager_async())

    async def init_sandbox_async(self):
        if self._manager is not None and self._manager._pool_initialized:
            return

        if self._is_volcengine_sandbox():
            from ms_enclave.sandbox.model import SandboxType, VolcengineSandboxConfig

            sandbox_type = SandboxType.VOLCENGINE
            sandbox_config = VolcengineSandboxConfig.model_validate(self._benchmark_meta.sandbox_config or {})
        else:
            from ms_enclave.sandbox.model import DockerSandboxConfig, SandboxType

            sandbox_type = self._normalize_sandbox_type(
                self._task_config.sandbox_type or SandboxType.DOCKER, SandboxType
            )
            sandbox_config = self._resolve_sandbox_config(sandbox_type, DockerSandboxConfig)

            if self._is_docker_sandbox(sandbox_type) and self.should_build_image(sandbox_config.image):
                logger.info(f'Building sandbox image: {sandbox_config.image}')
                build_context_path, dockerfile = self.get_build_context()
                self.build_docker_image(sandbox_config.image, path=build_context_path, dockerfile=dockerfile)
                logger.info(f'Sandbox image built: {sandbox_config.image}')

        sandbox_pool = await self._manager.initialize_pool(
            pool_size=self._pool_size, sandbox_type=sandbox_type, config=sandbox_config
        )

        logger.info(f'Sandbox pool initialized with {len(sandbox_pool)} sandboxes.')
        return

    def init_sandbox(self) -> Optional[str]:
        return AsyncioLoopRunner.run(self.init_sandbox_async())

    def should_build_image(self, image: str) -> bool:
        if not self._use_custom_image:
            return False

        from docker.client import DockerClient

        docker_client = DockerClient.from_env()
        avaliable_images = [tag for image in docker_client.images.list() for tag in image.tags]

        return image not in avaliable_images

    def build_docker_image(self, image: str, path: str, dockerfile: str = 'Dockerfile') -> Any:
        from docker.client import DockerClient
        docker_client = DockerClient.from_env()

        build_logs = docker_client.images.build(path=path, dockerfile=dockerfile, tag=image, rm=True)
        for log in build_logs[1]:
            if 'stream' in log:
                logger.info(f"{log['stream'].strip()}")
            elif 'error' in log:
                logger.error(f"{log['error']}")
        return build_logs[0]

    def get_build_context(self):
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dockerfile_path = os.path.join(current_dir, 'docker', 'Dockerfile')
        build_context_path = os.path.join(current_dir, 'docker')
        return build_context_path, dockerfile_path

    def _normalize_sandbox_type(self, sandbox_type: Union[str, Any], sandbox_type_enum: Any) -> Union[str, Any]:
        if isinstance(sandbox_type, str):
            try:
                return sandbox_type_enum(sandbox_type)
            except Exception:
                return sandbox_type
        return sandbox_type

    def _resolve_sandbox_config(self, sandbox_type: Union[str, Any], default_config_cls: Any):
        return default_config_cls.model_validate(self._benchmark_meta.sandbox_config)

    def _is_docker_sandbox(self, sandbox_type: Union[str, Any]) -> bool:
        sandbox_type_name = (sandbox_type.value if hasattr(sandbox_type, 'value') else str(sandbox_type)).lower()
        return sandbox_type_name == 'docker'


class SandboxMixin:
    """Sandbox mixin for sandboxed code execution."""

    def __init__(self, benchmark_meta: 'BenchmarkMeta', task_config: Optional['TaskConfig'] = None):
        self._benchmark_meta = benchmark_meta
        self._task_config = task_config

        self._backend: Optional[SandboxBackend] = None
        """Sandbox backend instance."""

        super().__init__()

    @property
    def use_sandbox(self) -> bool:
        """
        Return whether to use sandbox for the benchmark.
        """
        if not self._task_config:
            return False
        else:
            return self._task_config.use_sandbox

    @property
    def sandbox_manager(self) -> Optional['SandboxManager']:
        """Get the sandbox manager instance."""
        if isinstance(self._backend, EnclaveSandboxBackend):
            return self._backend._manager
        return None

    def _get_backend(self) -> SandboxBackend:
        if self._backend:
            return self._backend
        self._backend = EnclaveSandboxBackend(self._benchmark_meta, self._task_config)
        return self._backend

    @thread_safe
    def ensure_sandbox_ready(self) -> bool:
        """
        Ensure the sandbox loop, manager, and sandbox instance are initialized.
        This method is thread-safe and idempotent.
        """
        if not self.use_sandbox:
            return False

        backend = self._get_backend()
        if backend.is_ready():
            return True

        backend.start()
        return True

    def execute_code_in_sandbox(self,
                                code: Union[str, List[str]],
                                timeout: int = 60,
                                language: str = 'python') -> Dict[str, Any]:
        """Execute code in the sandbox."""
        if not self.ensure_sandbox_ready():
            logger.warning('Sandbox is not initialized.')
            return {'error': 'Sandbox is not initialized.'}

        return self._get_backend().execute(code, timeout, language)

    def sandbox_finalize(self, *args, **kwargs):
        """Finalize the sandbox manager."""
        if self._backend:
            self._backend.stop()
