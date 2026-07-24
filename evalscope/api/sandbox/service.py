"""SandboxService: process-level facade for ms_enclave sandbox managers.

Unifies the two historical code paths:

* ``CodeExecutionSandboxMixin.EnclaveCodeExecutionBackend`` – one manager per benchmark, pooled.
* ``EnclaveAgentEnvironment`` – one manager per process, per-sample containers.

Both are now thin wrappers around :class:`SandboxService`.  The service
caches managers keyed by ``(engine, manager_config)`` so the same
``base_url`` / Docker daemon is reused across benchmarks and agent
environments within a process.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import threading
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Coroutine, Dict, List, Optional, Set, Tuple, TypeVar

from evalscope.utils.function_utils import AsyncioLoopThread
from evalscope.utils.logger import get_logger
from .config_builder import build_sandbox_config
from .engine import SandboxEngine, get_enclave_types, resolve_engine

if TYPE_CHECKING:
    from ms_enclave.sandbox.manager import SandboxManager

logger = get_logger()
T = TypeVar('T')


async def _run_manager_operation(
    service: Optional['SandboxService'],
    operation: Coroutine[Any, Any, T],
) -> T:
    if service is None:
        return await operation
    return await service._run(operation)


# ---------------------------------------------------------------------------
# Handles: returned to callers instead of raw SandboxManager to keep the
# lifecycle consistent (pool vs per-sample).
# ---------------------------------------------------------------------------


class PoolHandle:
    """Handle for a pooled sandbox warmed up via ``manager.initialize_pool``.

    ``execute_tool_in_pool`` borrows a free sandbox from the pool, runs the
    tool, and returns it.  The pool itself is stopped when the parent
    :class:`SandboxService` shuts down.
    """

    def __init__(
        self,
        manager: 'SandboxManager',
        service: Optional['SandboxService'] = None,
    ) -> None:
        self._manager = manager
        self._service = service

    @property
    def manager(self) -> 'SandboxManager':
        return self._manager

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        return await _run_manager_operation(
            self._service,
            self._manager.execute_tool_in_pool(tool_name, parameters),
        )


class SandboxHandle:
    """Handle for a single per-sample container created via ``manager.create_sandbox``.

    ``close()`` calls ``manager.delete_sandbox`` and is idempotent.
    """

    def __init__(
        self,
        manager: 'SandboxManager',
        sandbox_id: str,
        service: Optional['SandboxService'] = None,
    ) -> None:
        self._manager = manager
        self._sandbox_id: Optional[str] = sandbox_id
        self._service = service

    @property
    def sandbox_id(self) -> Optional[str]:
        return self._sandbox_id

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        if self._sandbox_id is None:
            raise RuntimeError('SandboxHandle already closed')
        sandbox_id = self._sandbox_id
        return await _run_manager_operation(
            self._service,
            self._manager.execute_tool(sandbox_id, tool_name, parameters),
        )

    async def put_dir(self, source_dir: str | Path, target_dir: str) -> bool:
        """Copy a host directory into the sandbox via ms_enclave SandboxManager."""
        if self._sandbox_id is None:
            raise RuntimeError('SandboxHandle already closed')
        sandbox_id = self._sandbox_id
        return await _run_manager_operation(
            self._service,
            self._manager.put_dir(sandbox_id, source_dir, target_dir),
        )

    async def close(self) -> None:
        if self._sandbox_id is None:
            return
        sandbox_id = self._sandbox_id
        try:
            await _run_manager_operation(self._service, self._manager.delete_sandbox(sandbox_id))
            logger.debug(f'SandboxService: sandbox {sandbox_id} deleted.')
        except Exception as exc:
            logger.warning(f'SandboxService: error deleting sandbox {sandbox_id}: {exc}')
        finally:
            self._sandbox_id = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def _freeze(cfg: Optional[Dict[str, Any]]) -> str:
    """Produce a stable, hashable representation of a manager config dict."""
    try:
        return json.dumps(cfg or {}, sort_keys=True, default=str)
    except Exception:
        return repr(cfg or {})


class _ServicePhase(Enum):
    OPEN = 'open'
    CLOSING = 'closing'
    CLOSED = 'closed'


class SandboxService:
    """Process-level registry of ms_enclave ``SandboxManager`` instances.

    Access via :func:`get_sandbox_service`; the lazily created singleton is
    cleaned up explicitly after evaluation or through a process-level atexit
    fallback.
    """

    def __init__(self) -> None:
        # (engine, frozen-config) → started SandboxManager
        self._managers: Dict[Tuple[SandboxEngine, str], 'SandboxManager'] = {}
        # In-flight starts are shared by concurrent callers on the service loop.
        self._manager_starts: Dict[Tuple[SandboxEngine, str], asyncio.Task['SandboxManager']] = {}
        self._pool_starts: Dict[Tuple[SandboxEngine, str], asyncio.Task[None]] = {}
        self._runtime = AsyncioLoopThread(name='SandboxServiceLoop')
        self._state_lock = threading.Lock()
        self._phase = _ServicePhase.OPEN
        self._operations: Set[Future[Any]] = set()
        self._shutdown_future: Optional[Future[None]] = None

    # ------------------------------------------------------------------
    # Dedicated async runtime
    # ------------------------------------------------------------------

    async def _run(self, operation: Coroutine[Any, Any, T]) -> T:
        """Run a manager operation on the loop that owns its async resources."""
        future = self._submit_operation(operation)
        return await asyncio.shield(asyncio.wrap_future(future))

    def _run_sync(
        self,
        operation: Coroutine[Any, Any, T],
        timeout: Optional[float] = None,
    ) -> T:
        future = self._submit_operation(operation)
        return future.result(timeout=timeout)

    def _submit_operation(self, operation: Coroutine[Any, Any, T]) -> 'Future[T]':
        with self._state_lock:
            if self._phase is not _ServicePhase.OPEN:
                operation.close()
                raise RuntimeError(f'SandboxService is {self._phase.value} and cannot accept new work.')
            future = self._runtime.submit(operation)
            self._operations.add(future)
        future.add_done_callback(self._finish_operation)
        return future

    def _finish_operation(self, future: Future[Any]) -> None:
        with self._state_lock:
            self._operations.discard(future)

    @property
    def _accepting_work(self) -> bool:
        with self._state_lock:
            return self._phase is _ServicePhase.OPEN

    # ------------------------------------------------------------------
    # Manager cache
    # ------------------------------------------------------------------

    async def get_or_create_manager(
        self,
        engine: SandboxEngine,
        manager_config: Optional[Dict[str, Any]] = None,
    ) -> 'SandboxManager':
        return await self._run(self._get_or_create_manager(engine, manager_config))

    async def _get_or_create_manager(
        self,
        engine: SandboxEngine,
        manager_config: Optional[Dict[str, Any]] = None,
    ) -> 'SandboxManager':
        """Return a started manager for ``(engine, manager_config)``."""
        key = (engine, _freeze(manager_config))
        existing = self._managers.get(key)
        if existing is not None:
            return existing

        start_task = self._manager_starts.get(key)
        if start_task is None:
            start_task = asyncio.create_task(self._start_manager(key, engine, manager_config or {}))
            self._manager_starts[key] = start_task
            start_task.add_done_callback(lambda task: self._remove_manager_start(key, task))
        return await asyncio.shield(start_task)

    def _remove_manager_start(
        self,
        key: Tuple[SandboxEngine, str],
        task: asyncio.Task['SandboxManager'],
    ) -> None:
        if not task.cancelled():
            task.exception()
        if self._manager_starts.get(key) is task:
            self._manager_starts.pop(key, None)

    def _remove_pool_start(
        self,
        key: Tuple[SandboxEngine, str],
        task: asyncio.Task[None],
    ) -> None:
        if not task.cancelled():
            task.exception()
        if self._pool_starts.get(key) is task:
            self._pool_starts.pop(key, None)

    async def _start_manager(
        self,
        key: Tuple[SandboxEngine, str],
        engine: SandboxEngine,
        manager_config: Dict[str, Any],
    ) -> 'SandboxManager':
        manager = self._construct_manager(engine, manager_config)
        try:
            await manager.start()
        except BaseException:
            await self._stop_manager(manager, context='partially started manager')
            raise

        self._managers[key] = manager
        logger.info(
            f'SandboxService: manager started for engine={engine.value} '
            f'(total_managers={len(self._managers)}).'
        )
        return manager

    async def _initialize_pool(
        self,
        engine: SandboxEngine,
        manager: 'SandboxManager',
        pool_size: int,
        sandbox_config: Any,
    ) -> None:
        sandbox_type, _, _, _ = get_enclave_types(engine)
        pool = await manager.initialize_pool(pool_size=pool_size, sandbox_type=sandbox_type, config=sandbox_config)
        logger.info(f'SandboxService: pool initialized with {len(pool)} sandboxes (engine={engine.value}).')

    async def _stop_manager(self, manager: 'SandboxManager', *, context: str = 'manager') -> None:
        try:
            await manager.stop()
            logger.info(f'SandboxService: {context} stopped.')
        except Exception as exc:
            logger.warning(f'SandboxService: error stopping {context}: {exc}')
            cleanup_all = getattr(manager, 'cleanup_all_sandboxes', None)
            if not callable(cleanup_all):
                return
            try:
                await cleanup_all()
                logger.info(f'SandboxService: fallback cleanup completed for {context}.')
            except Exception as cleanup_exc:
                logger.warning(f'SandboxService: fallback cleanup failed for {context}: {cleanup_exc}')

    @staticmethod
    async def _cancel_tasks(tasks: List[asyncio.Task[Any]]) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _construct_manager(self, engine: SandboxEngine, manager_config: Dict[str, Any]) -> 'SandboxManager':
        _, _, manager_cls, manager_config_cls = get_enclave_types(engine)

        if manager_cls is not None:
            cfg = (
                manager_config_cls.model_validate(manager_config) if manager_config_cls is not None else manager_config
            )
            return manager_cls(config=cfg) if manager_config_cls is not None else manager_cls(**manager_config)

        # Default path: ms_enclave ``SandboxManagerFactory`` (covers docker).
        from ms_enclave.sandbox.manager import SandboxManagerFactory
        return SandboxManagerFactory.create_manager(**manager_config)

    # ------------------------------------------------------------------
    # Public APIs: pooled (CodeExecutionSandboxMixin) and per-sample (Agent env)
    # ------------------------------------------------------------------

    async def acquire_pool(
        self,
        engine: SandboxEngine,
        pool_size: int,
        sandbox_config: Any,
        manager_config: Optional[Dict[str, Any]] = None,
    ) -> PoolHandle:
        return await self._run(self._acquire_pool(engine, pool_size, sandbox_config, manager_config))

    async def _acquire_pool(
        self,
        engine: SandboxEngine,
        pool_size: int,
        sandbox_config: Any,
        manager_config: Optional[Dict[str, Any]] = None,
    ) -> PoolHandle:
        """Warm up (if needed) and return a pooled handle for ``engine``."""
        key = (engine, _freeze(manager_config))
        manager = await self._get_or_create_manager(engine, manager_config)
        if not getattr(manager, '_pool_initialized', False):
            start_task = self._pool_starts.get(key)
            if start_task is None:
                start_task = asyncio.create_task(self._initialize_pool(engine, manager, pool_size, sandbox_config))
                self._pool_starts[key] = start_task
                start_task.add_done_callback(lambda task: self._remove_pool_start(key, task))
            await asyncio.shield(start_task)
        return PoolHandle(manager, service=self)

    async def create_sandbox(
        self,
        engine: SandboxEngine,
        sandbox_config: Any,
        manager_config: Optional[Dict[str, Any]] = None,
    ) -> SandboxHandle:
        return await self._run(self._create_sandbox(engine, sandbox_config, manager_config))

    async def _create_sandbox(
        self,
        engine: SandboxEngine,
        sandbox_config: Any,
        manager_config: Optional[Dict[str, Any]] = None,
    ) -> SandboxHandle:
        """Create a single per-sample sandbox and return its handle."""
        manager = await self._get_or_create_manager(engine, manager_config)
        sandbox_type, _, _, _ = get_enclave_types(engine)
        sandbox_id = await manager.create_sandbox(sandbox_type, sandbox_config)
        logger.debug(f'SandboxService: sandbox {sandbox_id} created (engine={engine.value}).')
        return SandboxHandle(manager, sandbox_id, service=self)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown_all_async(self) -> None:
        future = self._begin_shutdown()
        if future is not None:
            await asyncio.shield(asyncio.wrap_future(future))
        await self._stop_runtime_async()

    def _begin_shutdown(self) -> Optional[Future[None]]:
        with self._state_lock:
            if self._shutdown_future is not None:
                return self._shutdown_future
            if self._phase is _ServicePhase.CLOSED:
                return None

            self._phase = _ServicePhase.CLOSING
            operations = list(self._operations)
            try:
                future = self._runtime.submit(self._shutdown_on_owner_loop(operations))
            except BaseException:
                self._phase = _ServicePhase.OPEN
                raise
            self._shutdown_future = future
        future.add_done_callback(self._finish_shutdown)
        return future

    def _finish_shutdown(self, future: Future[None]) -> None:
        self._runtime.stop()

    async def _shutdown_on_owner_loop(self, operations: List[Future[Any]]) -> None:
        """Drain accepted work and close every manager on the service loop."""
        await self._cancel_tasks(list(self._pool_starts.values()))
        await self._cancel_tasks(list(self._manager_starts.values()))
        self._pool_starts.clear()
        self._manager_starts.clear()

        if operations:
            await asyncio.gather(*(asyncio.wrap_future(future) for future in operations), return_exceptions=True)

        try:
            managers = list(self._managers.values())
            self._managers.clear()
            for manager in managers:
                await self._stop_manager(manager)
        finally:
            self._pool_starts.clear()
            self._manager_starts.clear()
            self._managers.clear()
            with self._state_lock:
                self._phase = _ServicePhase.CLOSED

    async def _stop_runtime_async(self) -> None:
        if self._runtime.owns(asyncio.get_running_loop()):
            self._runtime.stop()
            return
        await asyncio.to_thread(self._runtime.stop)

    def shutdown_all(self) -> None:
        """Synchronous wrapper around :meth:`shutdown_all_async`."""
        future = self._begin_shutdown()
        try:
            if future is not None:
                future.result(timeout=600)
        except Exception as exc:
            logger.warning(f'SandboxService: shutdown_all failed: {exc}')
        finally:
            if future is None or future.done():
                self._runtime.stop()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_SERVICE: Optional[SandboxService] = None
_SERVICE_LOCK = threading.Lock()


def get_sandbox_service() -> SandboxService:
    """Return the process-wide :class:`SandboxService` singleton."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None or not _SERVICE._accepting_work:
            _SERVICE = SandboxService()
        return _SERVICE


def shutdown_sandbox_service() -> None:
    """Shut down the existing sandbox service without creating one.

    This is safe to call from normal evaluation teardown.  If no sandbox was
    ever used, the singleton is still ``None`` and the function is a no-op.
    The atexit hook remains a last-resort fallback for callers that do not
    perform explicit teardown.
    """
    global _SERVICE
    with _SERVICE_LOCK:
        service = _SERVICE
        _SERVICE = None
    if service is not None:
        service.shutdown_all()


atexit.register(shutdown_sandbox_service)

# ---------------------------------------------------------------------------
# Convenience helpers used by CodeExecutionSandboxMixin / EnclaveAgentEnvironment
# ---------------------------------------------------------------------------


def build_and_acquire_pool_sync(
    engine: SandboxEngine,
    pool_size: int,
    sandbox_config_dict: Optional[Dict[str, Any]],
    manager_config: Optional[Dict[str, Any]] = None,
) -> PoolHandle:
    """Synchronous helper for :class:`CodeExecutionSandboxMixin`.

    Combines :func:`build_sandbox_config` and :meth:`SandboxService.acquire_pool`
    and drives them through the service-owned event loop.
    """
    service = get_sandbox_service()
    sandbox_config = build_sandbox_config(engine, sandbox_config_dict)

    return service._run_sync(service._acquire_pool(engine, pool_size, sandbox_config, manager_config))


__all__ = [
    'PoolHandle',
    'SandboxHandle',
    'SandboxService',
    'build_and_acquire_pool_sync',
    'get_sandbox_service',
    'shutdown_sandbox_service',
]
