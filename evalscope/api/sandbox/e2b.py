"""Bundled image mapping and local validation for the optional E2B engine."""

from __future__ import annotations

import functools
import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from pydantic import ValidationError

from .compat import validate_e2b_fields
from .engine import SandboxConfigError, SandboxEngine, get_enclave_types

if TYPE_CHECKING:
    from evalscope.config import SandboxTaskConfig

E2B_IMAGE_MAPPING_FILE = Path(__file__).with_name('e2b_image_mapping.json.gz')


@functools.lru_cache(maxsize=1)
def load_e2b_image_mapping() -> Dict[str, str]:
    """Load the packaged source-to-ACR image mapping once per process."""
    with gzip.open(E2B_IMAGE_MAPPING_FILE, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def with_e2b_image_mapping(manager_config: Dict[str, Any]) -> Dict[str, Any]:
    """Copy manager settings and inject the mapping without changing cache keys."""
    if 'image_mapping' in manager_config:
        return dict(manager_config)
    return {**manager_config, 'image_mapping': load_e2b_image_mapping()}


def check_e2b_task_config(sandbox: 'SandboxTaskConfig', *, external_agent: bool) -> None:
    """Validate E2B settings locally without changing the user's configuration."""
    if str(sandbox.engine).strip().lower() != SandboxEngine.E2B.value:
        return
    if not sandbox.enabled:
        raise SandboxConfigError("sandbox.engine='e2b' requires sandbox.enabled=True.")
    if external_agent:
        raise SandboxConfigError('The e2b sandbox engine does not support ExternalAgentConfig yet; use a native agent.')
    _, _, _, manager_config_cls = get_enclave_types(SandboxEngine.E2B)
    from ms_enclave.sandbox.boxes.e2b_connection import require_e2b

    require_e2b()
    try:
        manager_config_cls.model_validate(sandbox.manager_config or {})
    except ValidationError as exc:
        raise SandboxConfigError(f'Invalid e2b sandbox.manager_config: {exc}') from exc
    validate_e2b_fields(sandbox.default_config or {})
