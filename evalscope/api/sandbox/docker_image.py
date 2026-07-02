"""Small Docker image build/reuse helper.

This module is intentionally independent from the ms_enclave sandbox service:
benchmarks can use it to prepare local images, then pass the resulting tag to
their existing sandbox/environment layer.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, Dict, List

from evalscope.utils.logger import get_logger
from .config_builder import build_docker_image, normalize_docker_build_context, should_build_docker_image

logger = get_logger()


class DockerImageSpec(BaseModel):
    """Description of a local Docker build."""

    name_prefix: str
    context_dir: str
    dockerfile: str = 'Dockerfile'
    build_args: Dict[str, str] = Field(default_factory=dict)
    cache_key_parts: List[str] = Field(default_factory=list)
    force_rebuild: bool = False


class DockerImageResult(BaseModel):
    """Result returned by :class:`DockerImageBuilder`."""

    image_tag: str
    reused: bool
    context_hash: str


class DockerImageBuilder:
    """Build or reuse a local Docker image tagged from a content hash."""

    builder_version: str = 'v1'

    def build_or_reuse(self, spec: DockerImageSpec) -> DockerImageResult:
        context_dir, dockerfile = normalize_docker_build_context(spec.context_dir, spec.dockerfile)
        context_hash = hash_build_context(
            context_dir,
            cache_key_parts=[self.builder_version, *spec.cache_key_parts, *_build_args_cache_parts(spec.build_args)],
        )
        image_tag = f'{_sanitize_tag_part(spec.name_prefix)}:{context_hash[:24]}'
        should_build = spec.force_rebuild or should_build_docker_image(image_tag)
        if should_build:
            logger.info(f'Docker image {image_tag!r} not found or rebuild requested. Building from {context_dir} ...')
            build_docker_image_with_args(
                image=image_tag,
                path=context_dir,
                dockerfile=dockerfile,
                build_args=spec.build_args,
            )
            logger.info(f'Docker image built: {image_tag}')
        return DockerImageResult(image_tag=image_tag, reused=not should_build, context_hash=context_hash)


def hash_build_context(context_dir: str, *, cache_key_parts: List[str] | None = None) -> str:
    """Return a stable hash for regular files under ``context_dir``."""
    root = Path(context_dir)
    digest = hashlib.sha256()
    for part in cache_key_parts or []:
        digest.update(b'part\0')
        digest.update(str(part).encode('utf-8'))
        digest.update(b'\0')
    for file_path in sorted(path for path in root.rglob('*') if path.is_file() and not path.is_symlink()):
        rel = file_path.relative_to(root).as_posix()
        digest.update(b'file\0')
        digest.update(rel.encode('utf-8'))
        digest.update(b'\0')
        with file_path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        try:
            mode = file_path.stat().st_mode & 0o777
        except OSError:
            mode = 0
        digest.update(f'\0mode:{mode:o}'.encode('ascii'))
    return digest.hexdigest()


def build_docker_image_with_args(
    image: str,
    path: str,
    dockerfile: str = 'Dockerfile',
    build_args: Dict[str, str] | None = None,
) -> Any:
    """Build a Docker image with optional build args."""
    if not build_args:
        return build_docker_image(image=image, path=path, dockerfile=dockerfile)

    from docker.client import DockerClient

    docker_client = DockerClient.from_env()
    build_logs = docker_client.images.build(
        path=path,
        dockerfile=dockerfile,
        tag=image,
        rm=True,
        buildargs=build_args,
    )
    for log in build_logs[1]:
        if 'stream' in log:
            logger.info(log['stream'].strip())
        elif 'error' in log:
            logger.error(log['error'])
    return build_logs[0]


def _sanitize_tag_part(value: str) -> str:
    chars = []
    for ch in value.lower():
        if ch.isalnum() or ch in '._-/':
            chars.append(ch)
        else:
            chars.append('-')
    text = ''.join(chars).strip('.-/')
    return text or 'evalscope-image'


def _build_args_cache_parts(build_args: Dict[str, str]) -> List[str]:
    return [f'build_arg:{key}={value}' for key, value in sorted(build_args.items())]


__all__ = [
    'DockerImageBuilder',
    'DockerImageResult',
    'DockerImageSpec',
    'hash_build_context',
]
