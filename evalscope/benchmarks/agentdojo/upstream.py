"""Lazy, version-gated access to the upstream ``agentdojo`` package.

``load_upstream()`` is the only entry point that imports the package, so the benchmark stays importable when the
optional dependency is missing. Lookups go through ``importlib.metadata.version`` and ``importlib.import_module`` at
call time, so tests can monkeypatch either one (call ``load_upstream.cache_clear()`` between such tests).
"""

import functools
import importlib
import importlib.metadata
import importlib.resources
from typing import Any, Dict

import yaml
from pydantic import BaseModel, ConfigDict

from evalscope.benchmarks.agentdojo.constants import (
    AGENTDOJO_INSTALL_HINT,
    BENCHMARK_VERSION,
    SUITES,
    UPSTREAM_VERSION,
)

UPSTREAM_DISTRIBUTION = 'agentdojo'

# Deliberately excludes ``agent_pipeline.agent_pipeline``, which imports every LLM client SDK at module level.
_SUITES_MODULE = 'agentdojo.task_suite.load_suites'
_ATTACKS_MODULE = 'agentdojo.attacks.attack_registry'
_MODELS_MODULE = 'agentdojo.models'


class UpstreamUnavailableError(ImportError):
    """The upstream package is not installed or cannot be imported; the message carries the install hint."""


class UpstreamVersionError(ValueError):
    """The installed upstream version differs from ``UPSTREAM_VERSION``."""


class UpstreamDataError(ValueError):
    """Upstream data (task suites, default system message or attack injections) cannot be loaded or used."""


class UpstreamContext(BaseModel):
    """Handles to the upstream objects needed at runtime; built once per process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    suites: Dict[str, Any]  # suite name -> TaskSuite of BENCHMARK_VERSION, in SUITES order
    system_message: str  # system_messages.yaml['default']
    model_names: Dict[str, str]  # agentdojo.models.MODEL_NAMES (ordered)
    attacks: Dict[str, type]  # agentdojo.attacks.attack_registry.ATTACKS


def _unavailable(reason: str, exc: BaseException) -> UpstreamUnavailableError:
    return UpstreamUnavailableError(
        f'AgentDojo requires the upstream package {UPSTREAM_DISTRIBUTION}=={UPSTREAM_VERSION}, but {reason}: '
        f'{type(exc).__name__}: {exc}. Install it with: {AGENTDOJO_INSTALL_HINT}'
    )


def _check_version() -> None:
    # Read from distribution metadata, so an incompatible version never runs its import-time code.
    try:
        installed = importlib.metadata.version(UPSTREAM_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise _unavailable('it is not installed', exc) from exc
    if installed != UPSTREAM_VERSION:
        raise UpstreamVersionError(
            f'AgentDojo requires {UPSTREAM_DISTRIBUTION}=={UPSTREAM_VERSION} (expected version {UPSTREAM_VERSION}), '
            f'but version {installed} is installed. Install the pinned version with: {AGENTDOJO_INSTALL_HINT}'
        )


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise _unavailable(f'importing {name} failed', exc) from exc


def _load_suites(suites_module: Any) -> Dict[str, Any]:
    try:
        available = suites_module.get_suites(BENCHMARK_VERSION)
        missing = [name for name in SUITES if name not in available]
    except Exception as exc:
        raise UpstreamDataError(
            f'Failed to load AgentDojo task suites for benchmark version {BENCHMARK_VERSION}: '
            f'{type(exc).__name__}: {exc}'
        ) from exc
    if missing:
        raise UpstreamDataError(
            f'AgentDojo benchmark version {BENCHMARK_VERSION} is missing required task suites: {missing} '
            f'(expected {list(SUITES)}).'
        )
    return {name: available[name] for name in SUITES}


def _load_system_message() -> str:
    # Same source as upstream ``load_system_message``, without importing ``agent_pipeline.agent_pipeline``.
    try:
        path = importlib.resources.files(UPSTREAM_DISTRIBUTION) / 'data' / 'system_messages.yaml'
        message = yaml.safe_load(path.read_text(encoding='utf-8'))['default']
    except Exception as exc:
        raise UpstreamDataError(
            f'Failed to read the default AgentDojo system message for benchmark version {BENCHMARK_VERSION}: '
            f'{type(exc).__name__}: {exc}'
        ) from exc
    if not isinstance(message, str) or not message.strip():
        raise UpstreamDataError(
            f'The default AgentDojo system message for benchmark version {BENCHMARK_VERSION} is empty or not a string.'
        )
    return message


@functools.lru_cache(maxsize=1)
def load_upstream() -> UpstreamContext:
    """Check the upstream version, import the package and load the pinned task suites (cached per process).

    Raises:
        UpstreamUnavailableError: The package is not installed or fails to import.
        UpstreamVersionError: The installed version is not ``UPSTREAM_VERSION``.
        UpstreamDataError: Suites or the default system message cannot be loaded.
    """
    _check_version()
    suites_module = _import(_SUITES_MODULE)
    attacks_module = _import(_ATTACKS_MODULE)
    models_module = _import(_MODELS_MODULE)
    return UpstreamContext(
        suites=_load_suites(suites_module),
        system_message=_load_system_message(),
        model_names=dict(models_module.MODEL_NAMES),
        attacks=dict(attacks_module.ATTACKS),
    )
