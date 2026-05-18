"""Sandbox config builders and Docker image utilities.

These helpers turn dict-shaped configuration (e.g. from ``BenchmarkMeta.sandbox_config``
or ``TaskConfig.sandbox.default_config``) into typed ms_enclave config
objects and optionally build custom Docker images before the sandbox pool
warms up.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from evalscope.utils.logger import get_logger
from .engine import SandboxEngine, get_enclave_types

logger = get_logger()


def merge_sandbox_config_dicts(*sources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Shallow-merge sandbox config dicts left-to-right, later wins.

    ``None`` and empty dicts are skipped.  Used to combine the three layers:
    ``BenchmarkMeta.sandbox_config`` < ``TaskConfig.sandbox.default_config``
    < per-sample override.
    """
    merged: Dict[str, Any] = {}
    for src in sources:
        if src:
            merged.update(src)
    return merged


def build_sandbox_config(engine: SandboxEngine, cfg_dict: Optional[Dict[str, Any]]) -> Any:
    """Instantiate the engine-specific ``SandboxConfig`` from a dict."""
    _, config_cls, _, _ = get_enclave_types(engine)
    return config_cls.model_validate(cfg_dict or {})


def should_build_docker_image(image: str) -> bool:
    """Return True iff the local docker engine does not already have ``image``."""
    try:
        from docker.client import DockerClient
    except ImportError:
        logger.warning('docker SDK not installed; skipping image existence check.')
        return False

    try:
        docker_client = DockerClient.from_env()
        available = [tag for img in docker_client.images.list() for tag in img.tags]
    except Exception as exc:
        logger.warning(f'Unable to query docker images: {exc}')
        return False

    return image not in available


def build_docker_image(image: str, path: str, dockerfile: str = 'Dockerfile') -> Any:
    """Build a Docker image at ``path`` using the given ``dockerfile``."""
    from docker.client import DockerClient

    docker_client = DockerClient.from_env()
    build_logs = docker_client.images.build(path=path, dockerfile=dockerfile, tag=image, rm=True)
    for log in build_logs[1]:
        if 'stream' in log:
            logger.info(log['stream'].strip())
        elif 'error' in log:
            logger.error(log['error'])
    return build_logs[0]


def default_docker_build_context() -> tuple[str, str]:
    """Return the evalscope-bundled Dockerfile build context.

    Mirrors the historical layout in ``evalscope/api/mixin/docker/`` used by
    benchmarks that opt into the bundled image.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mixin_docker = os.path.normpath(os.path.join(current_dir, '..', 'mixin', 'docker'))
    dockerfile_path = os.path.join(mixin_docker, 'Dockerfile')
    return mixin_docker, dockerfile_path


__all__ = [
    'build_sandbox_config',
    'build_docker_image',
    'default_docker_build_context',
    'merge_sandbox_config_dicts',
    'should_build_docker_image',
]
