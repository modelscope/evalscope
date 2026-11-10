"""Central conversion rules from benchmark Docker settings to E2B settings."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import ValidationError

from .engine import SandboxConfigError, SandboxEngine, get_enclave_types

E2B_DEFAULT_USER = 'root'
SANDBOX_LIFETIME_MARGIN_S = 600


def memory_limit_to_mib(value: Any) -> int:
    """Convert a Docker memory size to MiB, rounding upward to at least one."""
    match = re.fullmatch(r'(\d+)([kmgKMG]?)', str(value).strip())
    if match is None:
        raise SandboxConfigError(f'Invalid memory_limit {value!r}; expected e.g. 512m or 4g.')
    factor = {'': 1, 'k': 1024, 'm': 1024**2, 'g': 1024**3}[match.group(2).lower()]
    return max(1, -(-int(match.group(1)) * factor // 1024**2))


_IGNORED_DOCKER_FIELDS = frozenset(
    {
        'command',
        'platform',
        'pull_progress',
        'pull_progress_interval',
        'remove_on_exit',
        'docker_executor_workers',
        'cap_add',
    }
)
_UNSUPPORTED_DOCKER_FIELDS = frozenset({'volumes', 'privileged', 'entrypoint', 'ports', 'extra_hosts', 'network'})
_FIELD_CONVERSIONS = {
    'network_enabled': ('allow_internet_access', bool),
    'memory_limit': ('memory_mb', memory_limit_to_mib),
    'cpu_limit': ('cpu_count', lambda value: max(1, math.ceil(float(value)))),
}


@lru_cache(maxsize=1)
def _docker_defaults() -> Dict[str, Any]:
    _, config_cls, _, _ = get_enclave_types(SandboxEngine.DOCKER)
    return {name: field.get_default(call_default_factory=True) for name, field in config_cls.model_fields.items()}


def _apply_e2b_rules(config: Dict[str, Any]) -> Dict[str, Any]:
    adapted, converted, rejected = {}, {}, []
    defaults = _docker_defaults()
    for key, value in config.items():
        if key in _IGNORED_DOCKER_FIELDS:
            continue
        if key in _UNSUPPORTED_DOCKER_FIELDS:
            if value != defaults[key]:
                rejected.append(key)
            continue
        if key in _FIELD_CONVERSIONS:
            target, convert = _FIELD_CONVERSIONS[key]
            try:
                converted[target] = convert(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SandboxConfigError(f'Invalid sandbox field {key}: {exc}') from exc
        else:
            adapted[key] = value
    if rejected:
        raise SandboxConfigError(f'Sandbox fields {sorted(rejected)} are not supported by the e2b engine.')
    for key, value in converted.items():
        adapted.setdefault(key, value)
    adapted.setdefault('user', E2B_DEFAULT_USER)
    return adapted


def adapt_sandbox_config(
    engine: SandboxEngine, config: Dict[str, Any], *, expected_duration: Optional[float] = None
) -> Dict[str, Any]:
    """Apply engine rules and bound a remote evaluation sandbox's lifetime."""
    if engine is not SandboxEngine.E2B:
        return dict(config)
    adapted = _apply_e2b_rules(config)
    if expected_duration is not None:
        lifetime = math.ceil(expected_duration) + SANDBOX_LIFETIME_MARGIN_S
        explicit = adapted.get('sandbox_timeout')
        adapted['sandbox_timeout'] = min(lifetime, int(explicit)) if explicit is not None else lifetime
    if not adapted.get('image') and not adapted.get('template'):
        raise SandboxConfigError(
            'The e2b engine needs an `image` from the benchmark or `template` in sandbox.default_config.'
        )
    return adapted


def validate_e2b_fields(config: Dict[str, Any]) -> None:
    """Validate partial task overrides before the benchmark supplies its image."""
    _, config_cls, _, _ = get_enclave_types(SandboxEngine.E2B)
    adapted = _apply_e2b_rules(config)
    # A placeholder satisfies the source invariant without mutating task overrides.
    if not adapted.get('image') and not adapted.get('template'):
        adapted['image'] = '__benchmark_image__'
    try:
        config_cls.model_validate(adapted)
    except ValidationError as exc:
        raise SandboxConfigError(f'Invalid e2b sandbox.default_config: {exc}') from exc
