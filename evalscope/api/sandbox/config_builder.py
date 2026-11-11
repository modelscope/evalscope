"""Sandbox config builders and Docker image utilities.

These helpers turn dict-shaped configuration (e.g. from ``BenchmarkMeta.sandbox_config``
or ``TaskConfig.sandbox.default_config``) into typed ms_enclave config
objects and optionally build custom Docker images before the sandbox pool
warms up.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import ValidationError

from evalscope.utils.argument_utils import get_secret_value
from evalscope.utils.logger import get_logger

from .compat import adapt_sandbox_config
from .engine import SandboxConfigError, SandboxEngine, get_enclave_types, resolve_engine

if TYPE_CHECKING:
    from evalscope.config import SandboxTaskConfig

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


def resolve_sandbox_config(defaults: Dict[str, Any], sandbox: Optional['SandboxTaskConfig']) -> Dict[str, Any]:
    """Merge task overrides and select Aliyun's base template for default Python execution."""
    overrides = sandbox.default_config if sandbox is not None else {}
    config = merge_sandbox_config_dicts(defaults, overrides)
    if (
        sandbox is None
        or not sandbox.enabled
        or resolve_engine(sandbox.engine) is not SandboxEngine.E2B
        or defaults.get('image') != 'python:3.11-slim'
        or config.get('template')
        or any(key in overrides for key in ('image', 'template'))
    ):
        return config

    _, _, _, manager_config_cls = get_enclave_types(SandboxEngine.E2B)
    manager = manager_config_cls.model_validate(sandbox.manager_config)
    image = defaults['image']
    if manager.provider == 'aliyun' and not any(key in manager.image_mapping for key in (image, f'docker.io/{image}')):
        config.pop('image')
        config['template'] = 'base'
    return config


def build_sandbox_config(
    engine: SandboxEngine, cfg_dict: Optional[Dict[str, Any]], *, expected_duration: Optional[float] = None
) -> Any:
    """Instantiate the engine-specific ``SandboxConfig`` from a dict."""
    _, config_cls, _, _ = get_enclave_types(engine)
    adapted = adapt_sandbox_config(engine, get_secret_value(cfg_dict or {}), expected_duration=expected_duration)
    try:
        return config_cls.model_validate(adapted)
    except ValidationError as exc:
        raise SandboxConfigError(f'Invalid {engine.value} sandbox config: {exc}') from exc


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


def normalize_docker_build_context(path: str, dockerfile: str = 'Dockerfile') -> tuple[str, str]:
    """Return absolute build context and dockerfile path relative to it."""
    build_ctx = os.path.abspath(path)
    dockerfile_path = dockerfile if os.path.isabs(dockerfile) else os.path.join(build_ctx, dockerfile)
    dockerfile_path = os.path.abspath(dockerfile_path)
    try:
        dockerfile_path = os.path.relpath(dockerfile_path, build_ctx)
    except ValueError:
        pass
    return build_ctx, dockerfile_path


def ensure_docker_image_built(
    image: str,
    path: str,
    dockerfile: str = 'Dockerfile',
    label: str = 'Docker image',
) -> bool:
    """Build ``image`` from ``path`` when it is missing from the local Docker daemon."""
    if not should_build_docker_image(image):
        return False

    build_ctx, dockerfile_path = normalize_docker_build_context(path, dockerfile)
    logger.info(f'{label} {image!r} not found. Building from {build_ctx} ...')
    build_docker_image(image, path=build_ctx, dockerfile=dockerfile_path)
    logger.info(f'{label} built: {image}')
    return True


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
    'ensure_docker_image_built',
    'merge_sandbox_config_dicts',
    'resolve_sandbox_config',
    'normalize_docker_build_context',
    'should_build_docker_image',
]
