"""Sandbox engine resolution.

Maps user-facing engine names (docker / volcengine / aliases) to the
concrete ms_enclave ``SandboxType``, ``SandboxConfig`` subclass and the
``SandboxManager`` class to instantiate.

This module is intentionally import-light: ms_enclave types are imported
lazily inside the helpers so that ``evalscope`` can be imported without the
sandbox extra.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from evalscope.config import TaskConfig


class SandboxConfigError(ValueError):
    """Invalid local sandbox settings detected before remote preparation."""


class SandboxEngine(str, Enum):
    """Supported sandbox engines."""

    DOCKER = 'docker'
    VOLCENGINE = 'volcengine'
    E2B = 'e2b'


_ALIASES = {
    'docker': SandboxEngine.DOCKER,
    'e2b': SandboxEngine.E2B,
    'volcengine': SandboxEngine.VOLCENGINE,
    'volcano': SandboxEngine.VOLCENGINE,
    'volc': SandboxEngine.VOLCENGINE,
}


def resolve_engine(value: Union[str, SandboxEngine, None]) -> SandboxEngine:
    """Normalise ``value`` into :class:`SandboxEngine`.

    Raises ``ValueError`` if the value cannot be resolved.  ``None`` defaults
    to :attr:`SandboxEngine.DOCKER` because it mirrors the historical
    ``TaskConfig.sandbox_type`` default.
    """
    if value is None:
        return SandboxEngine.DOCKER
    if isinstance(value, SandboxEngine):
        return value
    key = str(value).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f'Unknown sandbox engine: {value!r}. Supported: {sorted(set(_ALIASES))}')


def get_enclave_types(
    engine: SandboxEngine,
) -> Tuple[Any, Type[Any], Optional[Type[Any]], Optional[Type[Any]]]:
    """Return ``(SandboxType, SandboxConfigCls, ManagerCls, ManagerConfigCls)``.

    - ``ManagerCls`` is ``None`` when the default ``SandboxManagerFactory``
      should be used (docker case).
    - ``ManagerConfigCls`` is ``None`` when the manager takes raw ``**kwargs``.
    """
    if engine is SandboxEngine.DOCKER:
        from ms_enclave.sandbox.model import DockerSandboxConfig, SandboxType

        return SandboxType.DOCKER, DockerSandboxConfig, None, None

    if engine is SandboxEngine.VOLCENGINE:
        from ms_enclave.sandbox.manager import VolcengineSandboxManager
        from ms_enclave.sandbox.model import SandboxType, VolcengineSandboxConfig, VolcengineSandboxManagerConfig

        return (
            SandboxType.VOLCENGINE,
            VolcengineSandboxConfig,
            VolcengineSandboxManager,
            VolcengineSandboxManagerConfig,
        )

    if engine is SandboxEngine.E2B:
        try:
            from ms_enclave.sandbox.manager import E2BSandboxManager
            from ms_enclave.sandbox.model import E2BSandboxConfig, E2BSandboxManagerConfig, SandboxType
        except ImportError as exc:
            raise ImportError(
                "The 'e2b' sandbox engine requires ms-enclave with E2B support: pip install -U 'ms-enclave[e2b]'"
            ) from exc
        return SandboxType.E2B, E2BSandboxConfig, E2BSandboxManager, E2BSandboxManagerConfig

    raise ValueError(f'Unsupported sandbox engine: {engine!r}')


IMAGE_SANDBOX_ENGINES = frozenset({SandboxEngine.DOCKER, SandboxEngine.E2B})


def resolve_task_engine(task_config: Optional['TaskConfig']) -> SandboxEngine:
    """Resolve benchmark-owned image sandboxes, preserving disabled Docker defaults."""
    sandbox = task_config.sandbox if task_config is not None else None
    if sandbox is None:
        return SandboxEngine.DOCKER
    if not sandbox.enabled:
        if sandbox.engine == SandboxEngine.E2B or str(sandbox.engine).strip().lower() == 'e2b':
            raise SandboxConfigError("sandbox.engine='e2b' requires sandbox.enabled=True.")
        return SandboxEngine.DOCKER
    try:
        engine = resolve_engine(sandbox.engine)
    except ValueError as exc:
        raise SandboxConfigError(f'This benchmark needs an image-based sandbox engine (docker or e2b); {exc}') from exc
    if engine not in IMAGE_SANDBOX_ENGINES:
        raise SandboxConfigError(
            f'This benchmark needs an image-based sandbox engine (docker or e2b); '
            f'sandbox.engine={sandbox.engine!r} is not supported here.'
        )
    return engine


def task_sandbox_env_kwargs(
    task_config: Optional['TaskConfig'], *, docker_only_reason: Optional[str] = None
) -> Dict[str, Any]:
    """Return shared environment settings and enforce declared Docker-only needs."""
    engine = resolve_task_engine(task_config)
    if engine is SandboxEngine.DOCKER:
        return {'engine': engine}
    if docker_only_reason:
        raise SandboxConfigError(
            f'This benchmark does not support the {engine.value} sandbox engine yet: {docker_only_reason}'
        )
    return {'engine': engine, 'manager_config': dict(task_config.sandbox.manager_config or {})}


def sandbox_image_arch(engine: SandboxEngine) -> Optional[str]:
    """Return a remote engine's fixed image architecture, or defer to the host."""
    return 'x86_64' if engine is SandboxEngine.E2B else None


__all__ = [
    'SandboxEngine',
    'SandboxConfigError',
    'resolve_engine',
    'get_enclave_types',
    'IMAGE_SANDBOX_ENGINES',
    'resolve_task_engine',
    'task_sandbox_env_kwargs',
    'sandbox_image_arch',
]
