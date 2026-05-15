"""Unified sandbox service layer for evalscope.

This package consolidates the ms_enclave integration used by both
:class:`evalscope.api.mixin.sandbox_mixin.SandboxMixin` (pool-based execution
for code benchmarks) and the Agent environment (per-sample containers).

Public surface:

* :class:`SandboxEngine` / :func:`resolve_engine` – engine name handling
* :func:`build_sandbox_config` – build typed ms_enclave ``SandboxConfig``
* :class:`SandboxService` / :func:`get_sandbox_service` – manager lifecycle
* :class:`PoolHandle` / :class:`SandboxHandle` – caller-facing handles
"""

from .config_builder import (
    build_docker_image,
    build_sandbox_config,
    default_docker_build_context,
    merge_sandbox_config_dicts,
    should_build_docker_image,
)
from .engine import SandboxEngine, get_enclave_types, resolve_engine
from .service import PoolHandle, SandboxHandle, SandboxService, build_and_acquire_pool_sync, get_sandbox_service

__all__ = [
    'PoolHandle',
    'SandboxEngine',
    'SandboxHandle',
    'SandboxService',
    'build_and_acquire_pool_sync',
    'build_docker_image',
    'build_sandbox_config',
    'default_docker_build_context',
    'get_enclave_types',
    'get_sandbox_service',
    'merge_sandbox_config_dicts',
    'resolve_engine',
    'should_build_docker_image',
]
