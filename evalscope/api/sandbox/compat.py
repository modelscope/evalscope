"""Central conversion rules from benchmark Docker settings to E2B settings."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional

from pydantic import ValidationError

from .engine import SandboxConfigError, SandboxEngine, get_enclave_types

E2B_DEFAULT_USER = 'root'
SANDBOX_LIFETIME_MARGIN_S = 600


class FieldAction(str, Enum):
    """How a Docker field participates in E2B configuration."""

    KEEP = 'keep'
    DROP = 'drop'
    MAP = 'map'
    REJECT = 'reject'


@dataclass(frozen=True)
class FieldRule:
    """One immutable conversion rule."""

    action: FieldAction
    target: Optional[str] = None
    convert: Optional[Callable[[Any], Any]] = None


def memory_limit_to_mib(value: Any) -> int:
    """Convert a Docker memory size to MiB, rounding upward to at least one."""
    match = re.fullmatch(r'(\d+)([kmgKMG]?)', str(value).strip())
    if match is None:
        raise SandboxConfigError(f'Invalid memory_limit {value!r}; expected e.g. 512m or 4g.')
    factor = {'': 1, 'k': 1024, 'm': 1024**2, 'g': 1024**3}[match.group(2).lower()]
    return max(1, -(-int(match.group(1)) * factor // 1024**2))


E2B_FIELD_RULES: Mapping[str, FieldRule] = MappingProxyType(
    {
        **{key: FieldRule(FieldAction.KEEP) for key in ('image', 'working_dir', 'env_vars', 'tools_config')},
        **{
            key: FieldRule(FieldAction.DROP)
            for key in (
                'command',
                'platform',
                'pull_progress',
                'pull_progress_interval',
                'remove_on_exit',
                'docker_executor_workers',
                'cap_add',
            )
        },
        'network_enabled': FieldRule(FieldAction.MAP, 'allow_internet_access', bool),
        'memory_limit': FieldRule(FieldAction.MAP, 'memory_mb', memory_limit_to_mib),
        'cpu_limit': FieldRule(FieldAction.MAP, 'cpu_count', lambda v: max(1, math.ceil(float(v)))),
        **{
            key: FieldRule(FieldAction.REJECT)
            for key in ('volumes', 'privileged', 'entrypoint', 'ports', 'extra_hosts', 'network')
        },
    }
)


@lru_cache(maxsize=1)
def _docker_defaults() -> Dict[str, Any]:
    _, config_cls, _, _ = get_enclave_types(SandboxEngine.DOCKER)
    return {name: field.get_default(call_default_factory=True) for name, field in config_cls.model_fields.items()}


def _apply_e2b_rules(config: Dict[str, Any]) -> Dict[str, Any]:
    adapted, converted, rejected = {}, {}, []
    defaults = _docker_defaults()
    for key, value in config.items():
        rule = E2B_FIELD_RULES.get(key)
        if rule is None or rule.action is FieldAction.KEEP:
            adapted[key] = value
        elif rule.action is FieldAction.MAP:
            try:
                converted[rule.target] = rule.convert(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SandboxConfigError(f'Invalid sandbox field {key}: {exc}') from exc
        elif rule.action is FieldAction.REJECT and value != defaults[key]:
            rejected.append(key)
    if rejected:
        raise SandboxConfigError(
            f'Sandbox fields {sorted(rejected)} are not supported by the e2b engine; '
            'this benchmark does not support e2b yet.'
        )
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
