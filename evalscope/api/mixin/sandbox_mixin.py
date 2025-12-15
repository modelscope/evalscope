from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from evalscope.utils.function_utils import AsyncioLoopRunner, thread_safe
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from ms_enclave.sandbox.manager import SandboxManager

    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.config import TaskConfig

logger = get_logger()


class SandboxMixin:
    """Sandbox mixin for sandboxed code execution."""

    def __init__(self, benchmark_meta: 'BenchmarkMeta', task_config: Optional['TaskConfig'] = None):
        self._benchmark_meta = benchmark_meta
        self._task_config = task_config

        self._manager: Optional['SandboxManager'] = None
        """Sandbox manager instance."""

        self._pool_size: int = self._task_config.judge_worker_num if self._task_config else 1
        """Sandbox pool size."""

        self._use_custom_image: bool = False
        """Whether to use a custom sandbox image."""

        # Lazy init state
        self._initialized: bool = False

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
        return self._manager

    @thread_safe
    def ensure_sandbox_ready(self) -> bool:
        """
        Ensure the sandbox loop, manager, and sandbox instance are initialized.
        This method is thread-safe and idempotent.
        """
        if not self.use_sandbox:
            return False

        if self._initialized and self._manager and self._manager._pool_initialized:
            return True

        # Initialize manager and sandbox using the class-level runner
        self.init_sandbox_manager()
        self.init_sandbox()

        self._initialized = True
        return True

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
        # Process and log build output
        for log in build_logs[1]:  # build_logs[1] contains the build log generator
            if 'stream' in log:
                logger.info(f"{log['stream'].strip()}")
            elif 'error' in log:
                logger.error(f"{log['error']}")
        return build_logs[0]  # Return the built image

    def get_build_context(self):
        """Get the build context and Dockerfile path for building the sandbox image.

        Returns:
            Tuple[str, str]: A tuple containing the build context path and the Dockerfile path.
        """
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dockerfile_path = os.path.join(current_dir, 'docker', 'Dockerfile')
        build_context_path = os.path.join(current_dir, 'docker')
        return build_context_path, dockerfile_path

    async def init_sandbox_manager_async(self) -> Optional['SandboxManager']:
        """Initialize the sandbox manager asynchronously."""
        if self._manager is not None:
            return self._manager

        if not self.use_sandbox:
            return None

        from ms_enclave.sandbox.manager import SandboxManagerFactory

        manager_config = self._task_config.sandbox_manager_config or {}
        self._manager = SandboxManagerFactory.create_manager(**manager_config)

        await self._manager.start()
        logger.info('Sandbox manager initialized.')
        return self._manager

    def init_sandbox_manager(self) -> Optional['SandboxManager']:
        """Initialize the sandbox manager."""
        return AsyncioLoopRunner.run(self.init_sandbox_manager_async())

    async def init_sandbox_async(self):
        """Initialize the sandbox instance asynchronously."""
        if self._manager is not None and self._manager._pool_initialized:
            return

        if not self.use_sandbox:
            return

        from ms_enclave.sandbox.model import DockerSandboxConfig, SandboxType

        sandbox_config = DockerSandboxConfig.model_validate(self._benchmark_meta.sandbox_config)
        sandbox_type = self._task_config.sandbox_type or SandboxType.DOCKER

        if self.should_build_image(sandbox_config.image):
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
        """Initialize the sandbox instance."""
        return AsyncioLoopRunner.run(self.init_sandbox_async())

    def execute_code_in_sandbox(self,
                                code: Union[str, List[str]],
                                timeout: int = 60,
                                language: str = 'python') -> Dict[str, Any]:
        """Execute code in the sandbox."""
        # Lazy, thread-safe initialization
        if not self.ensure_sandbox_ready():
            logger.warning('Sandbox is not initialized.')
            return {'error': 'Sandbox is not initialized.'}

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
            result = await self._manager.execute_tool_in_pool(tool_name, parameters)
            return result

        # Execute in background loop via class-level runner
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
            # Avoid surfacing unexpected exceptions to callers
            logger.exception(f'Code execution in sandbox failed: {e!r}')
            return {'status': ExecutionStatus.ERROR, 'error': str(e), 'metadata': {'code': code, 'language': language}}

    def sandbox_finalize(self, *args, **kwargs):
        """Finalize the sandbox manager."""
        if self._manager:
            try:
                # Stop the manager but keep the shared loop alive
                AsyncioLoopRunner.run(self._manager.stop(), timeout=600)
                logger.info('Sandbox manager finalized.')
            except Exception as e:
                logger.warning(f'Error finalizing sandbox manager: {e}')
