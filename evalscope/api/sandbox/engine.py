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
from typing import Any, Optional, Tuple, Type, Union


class SandboxEngine(str, Enum):
    """Supported sandbox engines."""

    DOCKER = 'docker'
    VOLCENGINE = 'volcengine'


_ALIASES = {
    'docker': SandboxEngine.DOCKER,
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
    raise ValueError(f'Unknown sandbox engine: {value!r}. '
                     f'Supported: {sorted(set(_ALIASES))}')


def get_enclave_types(engine: SandboxEngine, ) -> Tuple[Any, Type[Any], Optional[Type[Any]], Optional[Type[Any]]]:
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

    raise ValueError(f'Unsupported sandbox engine: {engine!r}')


__all__ = ['SandboxEngine', 'resolve_engine', 'get_enclave_types']
