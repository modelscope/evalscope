import asyncio
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from ms_enclave.sandbox.manager import SandboxManager

    from evalscope.config import TaskConfig

logger = get_logger()


class SandboxMixin:
    """Sandbox mixin for sandboxed code execution."""

    def __init__(self, task_config: 'TaskConfig'):
        self._task_config = task_config

        self._manager: Optional['SandboxManager'] = None
        """Sandbox manager instance."""

        self._sandbox_id: Optional[str] = None
        """Sandbox ID."""

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        """Event loop for async operations."""

        # Initialize sandbox synchronously by running async methods
        if self.use_sandbox:
            self._loop = asyncio.new_event_loop()

            # Start the loop in a separate thread
            def run_loop():
                asyncio.set_event_loop(self._loop)
                self._loop.run_forever()

            self._loop_thread = threading.Thread(target=run_loop, daemon=True)
            self._loop_thread.start()

            # Wait for initialization
            future = asyncio.run_coroutine_threadsafe(self._async_init(), self._loop)
            future.result()

        super().__init__()

    async def _async_init(self):
        """Async initialization helper."""
        await self.init_sandbox_manager_async()
        await self.init_sandbox_async()

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

    @property
    def sandbox_id(self) -> Optional[str]:
        """Get the sandbox ID."""
        return self._sandbox_id

    async def init_sandbox_manager_async(self) -> Optional['SandboxManager']:
        """Initialize the sandbox manager asynchronously."""
        if self._manager is not None:
            return self._manager

        if not self.use_sandbox:
            return None

        from ms_enclave.sandbox.manager import HttpSandboxManager, LocalSandboxManager

        manager_config = self._task_config.sandbox_manager_config or {}
        if manager_config.get('base_url'):
            # Remote manager
            self._manager = HttpSandboxManager(**manager_config)
        else:
            # Local manager
            self._manager = LocalSandboxManager(**manager_config)

        await self._manager.start()
        logger.info('Sandbox manager initialized.')
        return self._manager

    def init_sandbox_manager(self) -> Optional['SandboxManager']:
        """Initialize the sandbox manager."""
        if self._manager is not None:
            return self._manager

        if not self.use_sandbox:
            return None

        # Use the dedicated loop if available
        if self._loop and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(self.init_sandbox_manager_async(), self._loop)
            return future.result()
        else:
            # Fallback for cases where no loop is available
            return asyncio.run(self.init_sandbox_manager_async())

    async def init_sandbox_async(self) -> Optional[str]:
        """Initialize the sandbox instance asynchronously."""
        if self._sandbox_id is not None:
            return self._sandbox_id

        if not self.use_sandbox:
            return None

        from ms_enclave.sandbox.model import DockerSandboxConfig, SandboxType

        sandbox_config = self._task_config.sandbox_config or DockerSandboxConfig(
            image='python:3.11-slim', tools_config={
                'shell_executor': {},
                'python_executor': {}
            }
        )
        sandbox_type = self._task_config.sandbox_type or SandboxType.DOCKER

        self._sandbox_id = await self._manager.create_sandbox(sandbox_type=sandbox_type, config=sandbox_config)

        sandbox_info = await self._manager.get_sandbox_info(self._sandbox_id)

        logger.info(f'Sandbox of type {sandbox_type} initialized. Info: {sandbox_info.model_dump(exclude_none=True)}')
        return self._sandbox_id

    def init_sandbox(self) -> Optional[str]:
        """Initialize the sandbox instance."""
        if self._sandbox_id is not None:
            return self._sandbox_id

        if not self.use_sandbox:
            return None

        # Use the dedicated loop if available
        if self._loop and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(self.init_sandbox_async(), self._loop)
            return future.result()
        else:
            # Fallback for cases where no loop is available
            return asyncio.run(self.init_sandbox_async())

    def execute_code_in_sandbox(self, code: str, timeout: int = 60, language: str = 'python') -> Dict[str, Any]:
        """Execute code in the sandbox."""
        if not self._sandbox_id or not self._manager:
            logger.warning('Sandbox is not initialized.')
            return {'error': 'Sandbox is not initialized.'}

        from ms_enclave.sandbox.model import ExecutionStatus, ToolResult

        async def _execute_async():
            if language.lower() == 'python':
                tool_name = 'python_executor'
                parameters = {'code': code, 'timeout': timeout}
                result = await self._manager.execute_tool(self._sandbox_id, tool_name, parameters)
            elif language.lower() == 'shell':
                tool_name = 'shell_executor'
                parameters = {'command': code, 'timeout': timeout}
                result = await self._manager.execute_tool(self._sandbox_id, tool_name, parameters)
            else:
                logger.warning(f"Unsupported language: {language}. Supported languages are 'python' and 'shell'.")
                result = ToolResult(
                    status=ExecutionStatus.ERROR,
                    tool_name='code_executor',
                    output=f"Unsupported language: {language}. Supported languages are 'python' and 'shell'."
                )
            return result

        # Use the dedicated loop if available
        if self._loop and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(_execute_async(), self._loop)
            result = future.result(timeout + 10)  # Add some buffer to the timeout
        else:
            # Fallback for cases where no loop is available
            result = asyncio.run(_execute_async())

        return result.model_dump(exclude_none=True)

    def sandbox_finalize(self, *args, **kwargs):
        """Finalize the sandbox manager."""
        if self._manager:
            try:
                if self._loop and not self._loop.is_closed():
                    # Stop the manager using the dedicated loop
                    future = asyncio.run_coroutine_threadsafe(self._manager.stop(), self._loop)
                    future.result(timeout=30)

                    # Stop the event loop
                    self._loop.call_soon_threadsafe(self._loop.stop)
                    if hasattr(self, '_loop_thread'):
                        self._loop_thread.join(timeout=5)

                logger.info('Sandbox manager finalized.')
            except Exception as e:
                logger.warning(f'Error finalizing sandbox manager: {e}')
