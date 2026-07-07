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

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.mixin.code_execution_sandbox_mixin import EnclaveCodeExecutionBackend
from evalscope.api.sandbox import (
    DockerImageResult,
    DockerImageSpec,
    PoolHandle,
    SandboxEngine,
    SandboxHandle,
    SandboxService,
    build_sandbox_config,
    ensure_docker_image_built,
    merge_sandbox_config_dicts,
    normalize_docker_build_context,
    resolve_engine,
    shutdown_sandbox_service,
)
from evalscope.config import SandboxTaskConfig, TaskConfig
from evalscope.run import shutdown_sandbox_service_if_enabled

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

    def test_custom_image_build_uses_adapter_context(self, tmp_path):
        dockerfile = tmp_path / 'Dockerfile.custom'
        dockerfile.write_text('FROM python:3.11-slim\n', encoding='utf-8')
        meta = BenchmarkMeta(
            name='custom_sandbox',
            dataset_id='dummy',
            sandbox_config={'image': 'custom-sandbox:latest'},
        )
        cfg = TaskConfig(sandbox={'enabled': True, 'engine': 'docker'})
        backend = EnclaveCodeExecutionBackend(
            benchmark_meta=meta,
            task_config=cfg,
            image_spec_provider=lambda: DockerImageSpec(
                name_prefix='custom-sandbox',
                context_dir=str(tmp_path),
                dockerfile=str(dockerfile),
                cache_key_parts=['sandbox', 'custom-sandbox:latest'],
            ),
        )

        with patch('evalscope.api.mixin.code_execution_sandbox_mixin.prepare_docker_image') as build_image, \
                patch('evalscope.api.mixin.code_execution_sandbox_mixin.build_and_acquire_pool_sync',
                      return_value=MagicMock()) as acquire_pool:
            build_image.return_value = DockerImageResult(
                image_tag='custom-sandbox:hash',
                reused=False,
                context_hash='hash',
            )
            backend.start()

        spec = build_image.call_args.args[0]
        assert spec.name_prefix == 'custom-sandbox'
        assert spec.context_dir == str(tmp_path)
        assert spec.dockerfile == str(dockerfile)
        assert spec.cache_key_parts == ['sandbox', 'custom-sandbox:latest']
        sandbox_config = acquire_pool.call_args.kwargs['sandbox_config_dict']
        assert sandbox_config['image'] == 'custom-sandbox:hash'

    def test_ensure_docker_image_built_normalizes_context(self, tmp_path):
        dockerfile = tmp_path / 'Dockerfile.custom'
        dockerfile.write_text('FROM python:3.11-slim\n', encoding='utf-8')

        build_ctx, dockerfile_name = normalize_docker_build_context(str(tmp_path), str(dockerfile))
        assert build_ctx == str(tmp_path.resolve())
        assert dockerfile_name == 'Dockerfile.custom'

        with patch('evalscope.api.sandbox.config_builder.should_build_docker_image', return_value=True), \
                patch('evalscope.api.sandbox.config_builder.build_docker_image') as build_image:
            built = ensure_docker_image_built(
                'custom-sandbox:latest',
                path=str(tmp_path),
                dockerfile=str(dockerfile),
                label='Sandbox image',
            )

        assert built is True
        build_image.assert_called_once_with(
            'custom-sandbox:latest',
            path=str(tmp_path.resolve()),
            dockerfile='Dockerfile.custom',
        )


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

    def test_shutdown_sandbox_service_noop_when_uncreated(self):
        from evalscope.api.sandbox import service as service_module

        old_service = service_module._SERVICE
        try:
            service_module._SERVICE = None
            shutdown_sandbox_service()
        finally:
            service_module._SERVICE = old_service

    def test_shutdown_sandbox_service_uses_existing_singleton(self):
        from evalscope.api.sandbox import service as service_module

        fake = MagicMock()
        old_service = service_module._SERVICE
        try:
            service_module._SERVICE = fake
            shutdown_sandbox_service()
            fake.shutdown_all.assert_called_once()
        finally:
            service_module._SERVICE = old_service


class TestSandboxServiceRunTeardown:

    def test_shutdown_skipped_when_sandbox_disabled(self):
        cfg = TaskConfig()
        with patch('evalscope.api.sandbox.shutdown_sandbox_service') as shutdown:
            shutdown_sandbox_service_if_enabled(cfg)
        shutdown.assert_not_called()

    def test_shutdown_called_when_sandbox_enabled(self):
        cfg = TaskConfig(sandbox={'enabled': True, 'engine': 'docker'})
        with patch('evalscope.api.sandbox.shutdown_sandbox_service') as shutdown:
            shutdown_sandbox_service_if_enabled(cfg)
        shutdown.assert_called_once_with()


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

    def test_sandbox_handle_put_dir_delegates_to_manager(self):
        manager = MagicMock()
        manager.put_dir = AsyncMock(return_value=True)
        handle = SandboxHandle(manager, 'sb-1')

        async def _run():
            return await handle.put_dir('/host/skills', '/skills')

        assert asyncio.run(_run()) is True
        manager.put_dir.assert_awaited_once_with('sb-1', '/host/skills', '/skills')


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
