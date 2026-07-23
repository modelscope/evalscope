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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Coroutine, Dict, List, Optional, Tuple, TypeVar

from evalscope.utils.logger import get_logger
from .config_builder import build_sandbox_config
from .engine import SandboxEngine, get_enclave_types, resolve_engine

if TYPE_CHECKING:
    from ms_enclave.sandbox.manager import SandboxManager

logger = get_logger()
T = TypeVar('T')


class _EventLoopThread:
    """Run loop-bound async resources on one long-lived event loop."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def active(self) -> bool:
        return self._loop is not None

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and not self._loop.is_closed():
                return self._loop

            loop = asyncio.new_event_loop()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                try:
                    loop.run_forever()
                finally:
                    loop.close()

            self._loop = loop
            self._thread = threading.Thread(target=_run, daemon=True, name=self._name)
            self._thread.start()
            return loop

    async def run(self, operation: Coroutine[Any, Any, T]) -> T:
        loop = self._ensure_started()
        if asyncio.get_running_loop() is loop:
            return await operation

        future = asyncio.run_coroutine_threadsafe(operation, loop)
        return await asyncio.wrap_future(future)

    def run_sync(
        self,
        operation: Coroutine[Any, Any, T],
        timeout: Optional[float] = None,
    ) -> T:
        loop = self._ensure_started()
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            raise RuntimeError('Cannot synchronously wait on the event loop thread.')

        future = asyncio.run_coroutine_threadsafe(operation, loop)
        return future.result(timeout=timeout)

    def stop(self, join_timeout: float = 5.0) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None

        if loop is None:
            return

        loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=join_timeout)


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


class SandboxService:
    """Process-level registry of ms_enclave ``SandboxManager`` instances.

    Access via :func:`get_sandbox_service`; the singleton is installed at
    import time and cleaned up through ``atexit``.
    """

    def __init__(self) -> None:
        # (engine, frozen-config) → started SandboxManager
        self._managers: Dict[Tuple[SandboxEngine, str], 'SandboxManager'] = {}
        # In-flight starts are shared by concurrent callers on the service loop.
        self._manager_starts: Dict[Tuple[SandboxEngine, str], asyncio.Task['SandboxManager']] = {}
        self._runtime = _EventLoopThread(name='SandboxServiceLoop')

    # ------------------------------------------------------------------
    # Dedicated async runtime
    # ------------------------------------------------------------------

    async def _run(self, operation: Coroutine[Any, Any, T]) -> T:
        """Run a manager operation on the loop that owns its async resources."""
        return await self._runtime.run(operation)

    def _run_sync(
        self,
        operation: Coroutine[Any, Any, T],
        timeout: Optional[float] = None,
    ) -> T:
        return self._runtime.run_sync(operation, timeout)

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
        try:
            return await start_task
        finally:
            if self._manager_starts.get(key) is start_task:
                self._manager_starts.pop(key, None)

    async def _start_manager(
        self,
        key: Tuple[SandboxEngine, str],
        engine: SandboxEngine,
        manager_config: Dict[str, Any],
    ) -> 'SandboxManager':
        manager = self._construct_manager(engine, manager_config)
        await manager.start()
        self._managers[key] = manager
        logger.info(
            f'SandboxService: manager started for engine={engine.value} '
            f'(total_managers={len(self._managers)}).'
        )
        return manager

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
        manager = await self._get_or_create_manager(engine, manager_config)
        if not getattr(manager, '_pool_initialized', False):
            sandbox_type, _, _, _ = get_enclave_types(engine)
            pool = await manager.initialize_pool(pool_size=pool_size, sandbox_type=sandbox_type, config=sandbox_config)
            logger.info(f'SandboxService: pool initialized with {len(pool)} sandboxes (engine={engine.value}).')
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
        try:
            await self._run(self._shutdown_all_async())
        finally:
            self._runtime.stop()

    async def _shutdown_all_async(self) -> None:
        managers = list(self._managers.values())
        self._managers.clear()
        for manager in managers:
            try:
                await manager.stop()
                logger.info('SandboxService: manager stopped.')
            except Exception as exc:
                logger.warning(f'SandboxService: error stopping manager: {exc}')
                cleanup_all = getattr(manager, 'cleanup_all_sandboxes', None)
                if not callable(cleanup_all):
                    continue
                try:
                    await cleanup_all()
                    logger.info('SandboxService: fallback sandbox cleanup completed.')
                except Exception as cleanup_exc:
                    logger.warning(f'SandboxService: fallback sandbox cleanup failed: {cleanup_exc}')

    def shutdown_all(self) -> None:
        """Synchronous wrapper around :meth:`shutdown_all_async`."""
        if not self._managers and not self._runtime.active:
            return
        try:
            self._run_sync(self._shutdown_all_async(), timeout=600)
        except Exception as exc:
            logger.warning(f'SandboxService: shutdown_all failed: {exc}')
        finally:
            self._runtime.stop()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_SERVICE: Optional[SandboxService] = None
_SERVICE_LOCK = threading.Lock()


def get_sandbox_service() -> SandboxService:
    """Return the process-wide :class:`SandboxService` singleton."""
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = SandboxService()
                atexit.register(_SERVICE.shutdown_all)
    return _SERVICE


def shutdown_sandbox_service() -> None:
    """Shut down the existing sandbox service without creating one.

    This is safe to call from normal evaluation teardown.  If no sandbox was
    ever used, the singleton is still ``None`` and the function is a no-op.
    The atexit hook remains a last-resort fallback for callers that do not
    perform explicit teardown.
    """
    if _SERVICE is None:
        return
    _SERVICE.shutdown_all()


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
