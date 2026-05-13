"""Unit tests for the unified sandbox service layer.

These tests focus on the pure-Python surface (engine resolution, config
builder, service-level bookkeeping) and do **not** require Docker or
ms_enclave managers to actually start.  Integration coverage lives in
``tests/agent/test_t2_environment.py`` (gated by Docker availability).
"""

from __future__ import annotations

import asyncio
import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from evalscope.api.sandbox import (
    PoolHandle,
    SandboxEngine,
    SandboxHandle,
    SandboxService,
    build_sandbox_config,
    merge_sandbox_config_dicts,
    resolve_engine,
)
from evalscope.config import SandboxTaskConfig, TaskConfig

# ===========================================================================
# Engine resolution
# ===========================================================================


class TestResolveEngine:

    def test_none_defaults_to_docker(self):
        assert resolve_engine(None) is SandboxEngine.DOCKER

    def test_enum_passthrough(self):
        assert resolve_engine(SandboxEngine.VOLCENGINE) is SandboxEngine.VOLCENGINE

    @pytest.mark.parametrize(
        'name',
        ['docker', 'Docker', ' DOCKER '],
    )
    def test_docker_aliases(self, name):
        assert resolve_engine(name) is SandboxEngine.DOCKER

    @pytest.mark.parametrize(
        'name',
        ['volcengine', 'volcano', 'volc', 'Volc'],
    )
    def test_volcengine_aliases(self, name):
        assert resolve_engine(name) is SandboxEngine.VOLCENGINE

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown sandbox engine'):
            resolve_engine('nonexistent')


# ===========================================================================
# Config builder
# ===========================================================================


class TestMergeSandboxConfigDicts:

    def test_later_wins(self):
        out = merge_sandbox_config_dicts({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
        assert out == {'a': 1, 'b': 3, 'c': 4}

    def test_none_and_empty_skipped(self):
        assert merge_sandbox_config_dicts(None, {}, {'x': 1}) == {'x': 1}

    def test_returns_new_dict(self):
        a = {'x': 1}
        out = merge_sandbox_config_dicts(a, {'y': 2})
        assert out is not a


class TestBuildSandboxConfig:

    def test_docker_builds_typed_config(self):
        pytest.importorskip('ms_enclave')
        from ms_enclave.sandbox.model import DockerSandboxConfig
        cfg = build_sandbox_config(
            SandboxEngine.DOCKER,
            {'image': 'python:3.11-slim', 'working_dir': '/ws'},
        )
        assert isinstance(cfg, DockerSandboxConfig)
        assert cfg.image == 'python:3.11-slim'


# ===========================================================================
# SandboxService manager caching
# ===========================================================================


class TestSandboxServiceCache:
    """Verify the manager cache keys by ``(engine, frozen(manager_config))``."""

    def _make_mock_manager(self) -> MagicMock:
        m = MagicMock()
        m.start = AsyncMock()
        m.stop = AsyncMock()
        return m

    def test_same_config_returns_cached_manager(self):
        service = SandboxService()
        fake = self._make_mock_manager()

        async def _run():
            with patch.object(service, '_construct_manager', return_value=fake) as ctor:
                a = await service.get_or_create_manager(SandboxEngine.DOCKER, {'base_url': 'x'})
                b = await service.get_or_create_manager(SandboxEngine.DOCKER, {'base_url': 'x'})
                assert a is b
                ctor.assert_called_once()

        asyncio.run(_run())

    def test_different_config_creates_new_manager(self):
        service = SandboxService()
        fake_1 = self._make_mock_manager()
        fake_2 = self._make_mock_manager()

        async def _run():
            managers = iter([fake_1, fake_2])
            with patch.object(service, '_construct_manager', side_effect=lambda *a, **kw: next(managers)):
                a = await service.get_or_create_manager(SandboxEngine.DOCKER, {'base_url': 'x'})
                b = await service.get_or_create_manager(SandboxEngine.DOCKER, {'base_url': 'y'})
                assert a is not b

        asyncio.run(_run())

    def test_shutdown_all_async_stops_and_clears(self):
        service = SandboxService()
        fake = self._make_mock_manager()

        async def _run():
            with patch.object(service, '_construct_manager', return_value=fake):
                await service.get_or_create_manager(SandboxEngine.DOCKER, {})
            assert service._managers  # populated
            await service.shutdown_all_async()
            fake.stop.assert_awaited_once()
            assert service._managers == {}

        asyncio.run(_run())


# ===========================================================================
# Handles
# ===========================================================================


class TestHandles:

    def test_pool_handle_execute_delegates(self):
        manager = MagicMock()
        manager.execute_tool_in_pool = AsyncMock(return_value='OK')
        handle = PoolHandle(manager)

        out = asyncio.run(handle.execute_tool('shell_executor', {'command': 'ls'}))
        assert out == 'OK'
        manager.execute_tool_in_pool.assert_awaited_once_with(
            'shell_executor', {'command': 'ls'}
        )

    def test_sandbox_handle_close_idempotent(self):
        manager = MagicMock()
        manager.delete_sandbox = AsyncMock()
        handle = SandboxHandle(manager, 'sb-1')

        async def _run():
            await handle.close()
            await handle.close()  # second call is no-op

        asyncio.run(_run())
        manager.delete_sandbox.assert_awaited_once_with('sb-1')

    def test_sandbox_handle_execute_after_close_raises(self):
        manager = MagicMock()
        manager.delete_sandbox = AsyncMock()
        handle = SandboxHandle(manager, 'sb-1')

        async def _run():
            await handle.close()
            await handle.execute_tool('x', {})

        with pytest.raises(RuntimeError, match='already closed'):
            asyncio.run(_run())


# ===========================================================================
# TaskConfig.sandbox ← legacy fields (legacy → nested one-way fold)
# ===========================================================================


class TestTaskConfigSandboxBridge:

    def test_legacy_only_projects_to_nested(self):
        cfg = TaskConfig(
            use_sandbox=True,
            sandbox_type='volcengine',
            sandbox_manager_config={'key': 'val'},
        )
        assert isinstance(cfg.sandbox, SandboxTaskConfig)
        assert cfg.sandbox.enabled is True
        assert cfg.sandbox.engine == 'volcengine'
        assert cfg.sandbox.manager_config == {'key': 'val'}

    def test_nested_does_not_mirror_back_to_legacy(self):
        # After the mirror was removed, legacy fields keep whatever the user
        # passed in (defaults here) and are NOT synchronised with `sandbox`.
        cfg = TaskConfig(sandbox={'enabled': True, 'engine': 'volc', 'manager_config': {'k': 'v'}})
        assert cfg.sandbox.enabled is True
        assert cfg.sandbox.engine == 'volc'
        assert cfg.sandbox.manager_config == {'k': 'v'}
        # Legacy fields stay at their defaults (not mirrored).
        assert cfg.use_sandbox is False
        assert cfg.sandbox_type == 'docker'
        assert cfg.sandbox_manager_config == {}

    def test_both_set_nested_wins(self):
        # Nested `sandbox` wins; legacy fields remain untouched inputs.
        cfg = TaskConfig(
            use_sandbox=False,  # contradicts the nested value below
            sandbox_type='docker',
            sandbox={'enabled': True, 'engine': 'volcengine', 'manager_config': {'k': 'v'}},
        )
        assert cfg.sandbox.enabled is True
        assert cfg.sandbox.engine == 'volcengine'
        assert cfg.sandbox.manager_config == {'k': 'v'}
        # Legacy fields are NOT mirrored from the nested object.
        assert cfg.use_sandbox is False
        assert cfg.sandbox_type == 'docker'

    def test_default_config_is_carried_through(self):
        cfg = TaskConfig(sandbox={
            'enabled': True,
            'engine': 'docker',
            'default_config': {'image': 'my:img'},
        })
        assert cfg.sandbox.default_config == {'image': 'my:img'}
