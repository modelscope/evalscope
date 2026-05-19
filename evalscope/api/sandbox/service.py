"""SandboxService: process-level facade for ms_enclave sandbox managers.

Unifies the two historical code paths:

* ``SandboxMixin.EnclaveSandboxBackend`` – one manager per benchmark, pooled.
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from evalscope.utils.function_utils import AsyncioLoopRunner
from evalscope.utils.logger import get_logger
from .config_builder import build_sandbox_config
from .engine import SandboxEngine, get_enclave_types, resolve_engine

if TYPE_CHECKING:
    from ms_enclave.sandbox.manager import SandboxManager

logger = get_logger()

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

    def __init__(self, manager: 'SandboxManager') -> None:
        self._manager = manager

    @property
    def manager(self) -> 'SandboxManager':
        return self._manager

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        return await self._manager.execute_tool_in_pool(tool_name, parameters)


class SandboxHandle:
    """Handle for a single per-sample container created via ``manager.create_sandbox``.

    ``close()`` calls ``manager.delete_sandbox`` and is idempotent.
    """

    def __init__(self, manager: 'SandboxManager', sandbox_id: str) -> None:
        self._manager = manager
        self._sandbox_id: Optional[str] = sandbox_id

    @property
    def sandbox_id(self) -> Optional[str]:
        return self._sandbox_id

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        if self._sandbox_id is None:
            raise RuntimeError('SandboxHandle already closed')
        return await self._manager.execute_tool(self._sandbox_id, tool_name, parameters)

    async def close(self) -> None:
        if self._sandbox_id is None:
            return
        try:
            await self._manager.delete_sandbox(self._sandbox_id)
            logger.debug(f'SandboxService: sandbox {self._sandbox_id} deleted.')
        except Exception as exc:
            logger.warning(f'SandboxService: error deleting sandbox {self._sandbox_id}: {exc}')
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
        # threading.Lock guards the cache itself + per-key startup events.
        # We deliberately do NOT use asyncio.Lock here: the SandboxService
        # is a process-level singleton and is now reachable from many event
        # loops (one per worker thread, see AsyncioLoopRunner). asyncio.Lock
        # binds to the loop it's first awaited on and would raise
        # "attached to a different event loop" from the second loop onward.
        self._thread_lock = threading.Lock()
        # Per-key threading.Event used to coordinate concurrent first-time
        # manager.start() calls without holding the cache lock across await.
        self._manager_starting: Dict[Tuple[SandboxEngine, str], threading.Event] = {}

    # ------------------------------------------------------------------
    # Manager cache
    # ------------------------------------------------------------------

    async def get_or_create_manager(
        self,
        engine: SandboxEngine,
        manager_config: Optional[Dict[str, Any]] = None,
    ) -> 'SandboxManager':
        """Return a started manager for ``(engine, manager_config)``; create if needed.

        Multi-loop safe: the cache lookup, manager construction, and the
        startup-coordination Event are guarded by ``threading.Lock`` (not
        asyncio.Lock). ``manager.start()`` is awaited *without* holding any
        lock so concurrent loops can't deadlock each other.
        """
        key = (engine, _freeze(manager_config))

        # Fast path: already-cached, ready manager.
        existing = self._managers.get(key)
        if existing is not None:
            return existing

        # Slow path: this thread either becomes the starter or waits for it.
        with self._thread_lock:
            existing = self._managers.get(key)
            if existing is not None:
                return existing

            starting_event = self._manager_starting.get(key)
            if starting_event is not None:
                # Another thread is in the middle of starting this manager.
                is_starter = False
            else:
                starting_event = threading.Event()
                self._manager_starting[key] = starting_event
                is_starter = True
                # Construct the manager object synchronously while the lock is held;
                # this is cheap (no IO) and ensures only one is ever built per key.
                manager = self._construct_manager(engine, manager_config or {})

        if not is_starter:
            # Block (off-loop) until the starter finishes, then read the result.
            await asyncio.to_thread(starting_event.wait)
            cached = self._managers.get(key)
            if cached is None:
                raise RuntimeError(f'SandboxService: peer failed to start manager for {engine.value}')
            return cached

        # Starter path: run the (potentially long) start() outside any lock.
        try:
            await manager.start()
            with self._thread_lock:
                self._managers[key] = manager
            logger.info(
                f'SandboxService: manager started for engine={engine.value} '
                f'(total_managers={len(self._managers)}).'
            )
            return manager
        finally:
            with self._thread_lock:
                self._manager_starting.pop(key, None)
            starting_event.set()

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
    # Public APIs: pooled (SandboxMixin) and per-sample (Agent env)
    # ------------------------------------------------------------------

    async def acquire_pool(
        self,
        engine: SandboxEngine,
        pool_size: int,
        sandbox_config: Any,
        manager_config: Optional[Dict[str, Any]] = None,
    ) -> PoolHandle:
        """Warm up (if needed) and return a pooled handle for ``engine``."""
        manager = await self.get_or_create_manager(engine, manager_config)
        if not getattr(manager, '_pool_initialized', False):
            sandbox_type, _, _, _ = get_enclave_types(engine)
            pool = await manager.initialize_pool(pool_size=pool_size, sandbox_type=sandbox_type, config=sandbox_config)
            logger.info(f'SandboxService: pool initialized with {len(pool)} sandboxes (engine={engine.value}).')
        return PoolHandle(manager)

    async def create_sandbox(
        self,
        engine: SandboxEngine,
        sandbox_config: Any,
        manager_config: Optional[Dict[str, Any]] = None,
    ) -> SandboxHandle:
        """Create a single per-sample sandbox and return its handle."""
        manager = await self.get_or_create_manager(engine, manager_config)
        sandbox_type, _, _, _ = get_enclave_types(engine)
        sandbox_id = await manager.create_sandbox(sandbox_type, sandbox_config)
        logger.debug(f'SandboxService: sandbox {sandbox_id} created (engine={engine.value}).')
        return SandboxHandle(manager, sandbox_id)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown_all_async(self) -> None:
        managers = list(self._managers.values())
        self._managers.clear()
        for manager in managers:
            try:
                await manager.stop()
                logger.info('SandboxService: manager stopped.')
            except Exception as exc:
                logger.warning(f'SandboxService: error stopping manager: {exc}')

    def shutdown_all(self) -> None:
        """Synchronous wrapper around :meth:`shutdown_all_async`."""
        if not self._managers:
            return
        try:
            AsyncioLoopRunner.run(self.shutdown_all_async(), timeout=600)
        except Exception as exc:
            logger.warning(f'SandboxService: shutdown_all failed: {exc}')


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


# ---------------------------------------------------------------------------
# Convenience helpers used by SandboxMixin / EnclaveAgentEnvironment
# ---------------------------------------------------------------------------


def build_and_acquire_pool_sync(
    engine: SandboxEngine,
    pool_size: int,
    sandbox_config_dict: Optional[Dict[str, Any]],
    manager_config: Optional[Dict[str, Any]] = None,
) -> PoolHandle:
    """Synchronous helper for :class:`SandboxMixin`.

    Combines :func:`build_sandbox_config` and :meth:`SandboxService.acquire_pool`
    and drives them through the shared :class:`AsyncioLoopRunner`.
    """
    service = get_sandbox_service()
    sandbox_config = build_sandbox_config(engine, sandbox_config_dict)

    async def _run() -> PoolHandle:
        return await service.acquire_pool(engine, pool_size, sandbox_config, manager_config)

    return AsyncioLoopRunner.run(_run())


__all__ = [
    'PoolHandle',
    'SandboxHandle',
    'SandboxService',
    'build_and_acquire_pool_sync',
    'get_sandbox_service',
]
