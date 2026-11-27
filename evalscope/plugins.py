# Copyright (c) Alibaba, Inc. and its affiliates.
"""Opt-in loading of third-party plugins discovered through Python entry points.

A plugin is an installed distribution that declares a module in the
``evalscope.plugins`` entry point group::

    [project.entry-points."evalscope.plugins"]
    company-audit = "company_audit.evalscope_plugin"

Importing that module runs the existing ``@register_*`` decorators (model APIs,
benchmarks, metrics, perf API protocols, perf datasets, ...). Nothing is imported
unless the plugin is enabled by name, e.g. ``evalscope eval --plugin company-audit``
or ``TaskConfig(plugins=['company-audit'])``.
"""

import argparse
import importlib.metadata
import sys
import threading
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple, Union

from pydantic import BaseModel, Field

from evalscope.utils.logger import get_logger

logger = get_logger()

ENTRY_POINT_GROUP = 'evalscope.plugins'

_MISSING = object()
_lock = threading.RLock()
_loaded: Dict[str, 'PluginInfo'] = {}


class PluginInfo(BaseModel):
    """Identity of an installed plugin and the names it registered once loaded."""

    name: str
    """Entry point name used to enable the plugin."""

    module: str
    """Entry point value, i.e. the module imported to register the plugin's components."""

    distribution: Optional[str] = None
    """Name of the installed distribution that provides the entry point."""

    version: Optional[str] = None
    """Version of that distribution."""

    registrations: Dict[str, List[str]] = Field(default_factory=dict)
    """Names registered by the plugin, keyed by registry kind (e.g. ``model_api``, ``perf_api``)."""

    def describe(self) -> str:
        """Return a one-line identity for logs and error messages."""
        source = f'{self.distribution} {self.version}' if self.distribution else 'unknown distribution'
        return f"plugin '{self.name}' ({source}, module '{self.module}')"


class PluginError(RuntimeError):
    """Base class for plugin discovery and loading errors."""


class PluginNotFoundError(PluginError):
    """Raised when an enabled plugin name is not provided by any installed distribution."""


class PluginLoadError(PluginError):
    """Raised when an enabled plugin cannot be imported."""


class PluginConflictError(PluginLoadError):
    """Raised when a plugin registers a name that is already registered."""


def add_plugin_argument(parser: argparse.ArgumentParser) -> None:
    """Add the repeatable ``--plugin`` option shared by the eval, perf and service commands."""
    parser.add_argument(
        '--plugin',
        dest='plugins',
        metavar='NAME',
        action='extend',
        nargs='+',
        default=None,
        help=f'Enable an installed plugin by its entry point name in the "{ENTRY_POINT_GROUP}" group. '
        'Repeat the option or list several names to enable more than one.',
    )


def normalize_plugin_names(names: Union[None, str, Iterable[Any]]) -> List[str]:
    """Return stripped, de-duplicated plugin names in their original order.

    Raises:
        ValueError: if a name is empty or not a string.
    """
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    result: List[str] = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'Plugin names must be non-empty strings, got {name!r}.')
        name = name.strip()
        if name not in result:
            result.append(name)
    return result


def available_plugins() -> List[PluginInfo]:
    """List installed plugins without importing them."""
    return sorted((_plugin_info(ep) for ep in _entry_points()), key=lambda info: info.name)


def loaded_plugins() -> List[PluginInfo]:
    """List the plugins loaded in this process, in load order."""
    with _lock:
        return [info.model_copy(deep=True) for info in _loaded.values()]


def load_plugins(names: Union[None, str, Iterable[str]]) -> List[PluginInfo]:
    """Import the named plugins once per process and return their identities.

    Loading is explicit: only the named entry points are imported. A plugin that is
    already loaded is not imported again, so callers may pass the same names on every
    run (the service does this in each spawned worker).

    Raises:
        PluginNotFoundError: a name is not provided by any installed distribution.
        PluginConflictError: a plugin registers a name that is already registered.
        PluginLoadError: a plugin fails to import or its name is ambiguous.
    """
    names = normalize_plugin_names(names)
    if not names:
        return []
    with _lock:
        pending = [name for name in names if name not in _loaded]
        if pending:
            entry_points = _entry_points_by_name()
            for name in pending:
                _loaded[name] = _load_one(name, entry_points)
        return [_loaded[name].model_copy(deep=True) for name in names]


def plugin_records(names: Union[None, str, Iterable[str]]) -> List[Dict[str, Any]]:
    """Return JSON-ready identities of the named plugins for run records."""
    return [info.model_dump(mode='json') for info in load_plugins(names)]


# --- internals -------------------------------------------------------------


def _entry_points() -> List[importlib.metadata.EntryPoint]:
    return list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))


def _plugin_info(entry_point: importlib.metadata.EntryPoint) -> PluginInfo:
    dist = getattr(entry_point, 'dist', None)
    return PluginInfo(
        name=entry_point.name,
        module=entry_point.value,
        distribution=dist.name if dist is not None else None,
        version=dist.version if dist is not None else None,
    )


def _entry_points_by_name() -> Dict[str, List[importlib.metadata.EntryPoint]]:
    grouped: Dict[str, List[importlib.metadata.EntryPoint]] = {}
    for entry_point in _entry_points():
        grouped.setdefault(entry_point.name, []).append(entry_point)
    return grouped


def _select_entry_point(
    name: str, entry_points: Dict[str, List[importlib.metadata.EntryPoint]]
) -> importlib.metadata.EntryPoint:
    candidates = entry_points.get(name, [])
    if not candidates:
        installed = ', '.join(sorted(entry_points)) or 'none'
        raise PluginNotFoundError(
            f"Plugin '{name}' is not installed. Installed plugins: {installed}. "
            f"Plugins are discovered from the '{ENTRY_POINT_GROUP}' entry point group."
        )
    if len(candidates) > 1:
        sources = '; '.join(_plugin_info(ep).describe() for ep in candidates)
        raise PluginLoadError(f"Plugin name '{name}' is provided by more than one distribution: {sources}.")
    return candidates[0]


def _tracked_registries() -> List[Tuple[str, MutableMapping[str, Any]]]:
    """Registries a plugin may write to, with the kind label used in records."""
    from evalscope.api import registry

    tracked: List[Tuple[str, MutableMapping[str, Any]]] = [
        ('benchmark', registry.BENCHMARK_REGISTRY),
        ('model_api', registry.MODEL_APIS),
        ('metric', registry.METRIC_REGISTRY),
        ('filter', registry.FILTER_REGISTRY),
        ('aggregation', registry.AGGREGATION_REGISTRY),
        ('evaluator', registry.EVALUATOR_REGISTRY),
        ('agent_strategy', registry.STRATEGY_REGISTRY),
        ('agent_environment', registry.ENVIRONMENT_REGISTRY),
        ('agent_runner', registry.RUNNER_REGISTRY),
        ('agent_tool', registry.AGENT_TOOL_REGISTRY),
    ]
    try:
        # Importing the package registers the built-in perf plugins first, so that a
        # plugin re-registering one of their names is reported instead of silently winning.
        import evalscope.perf.plugin  # noqa: F401
        from evalscope.perf.plugin.registry import ApiRegistry, DatasetRegistry
    except ImportError as e:
        logger.debug(f'Perf registries are unavailable for plugin tracking: {e}')
    else:
        tracked.append(('perf_api', ApiRegistry._registry))
        tracked.append(('perf_dataset', DatasetRegistry._registry))
    return tracked


def _snapshot(tracked: List[Tuple[str, MutableMapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    # dict.copy reads the backing storage directly, so lazy registries are not materialized.
    return {kind: dict.copy(mapping) for kind, mapping in tracked}


def _rollback(tracked: List[Tuple[str, MutableMapping[str, Any]]], before: Dict[str, Dict[str, Any]]) -> None:
    for kind, mapping in tracked:
        snapshot = before[kind]
        for key in [key for key in dict.keys(mapping) if key not in snapshot]:
            dict.__delitem__(mapping, key)
        for key, value in snapshot.items():
            if dict.get(mapping, key, _MISSING) is not value:
                dict.__setitem__(mapping, key, value)


def _overridden_names(
    tracked: List[Tuple[str, MutableMapping[str, Any]]], before: Dict[str, Dict[str, Any]]
) -> List[str]:
    overridden = []
    for kind, mapping in tracked:
        for key, value in before[kind].items():
            if dict.get(mapping, key, _MISSING) is not value:
                overridden.append(f"{kind} '{key}'")
    return overridden


def _registered_names(
    tracked: List[Tuple[str, MutableMapping[str, Any]]], before: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    registrations: Dict[str, List[str]] = {}
    for kind, mapping in tracked:
        added = []
        for key in dict.keys(mapping):
            if key in before[kind]:
                continue
            # A plugin that subclasses a built-in adapter imports its module, which registers
            # the built-in benchmark on the way; that name does not belong to the plugin.
            adapter = getattr(dict.get(mapping, key), 'data_adapter', None)
            if kind == 'benchmark' and getattr(adapter, '__module__', '').startswith('evalscope.benchmarks.'):
                continue
            added.append(key)
        if added:
            registrations[kind] = sorted(added)
    return registrations


def _discard_modules(root: str, before: set) -> None:
    """Drop modules of a failed plugin so that a later attempt re-runs its registrations."""
    for module_name in [m for m in sys.modules if m not in before]:
        if module_name == root or module_name.startswith(f'{root}.'):
            sys.modules.pop(module_name, None)


def _load_one(name: str, entry_points: Dict[str, List[importlib.metadata.EntryPoint]]) -> PluginInfo:
    from evalscope.api.registry import BENCHMARK_REGISTRY, RegistrationConflictError

    entry_point = _select_entry_point(name, entry_points)
    info = _plugin_info(entry_point)
    tracked = _tracked_registries()
    root_module = entry_point.module.split('.')[0]
    # Hold the benchmark registry lock so concurrent lazy loads of built-in adapters do
    # not interleave with this import and get attributed to (or rolled back with) it.
    with BENCHMARK_REGISTRY._lock:
        before = _snapshot(tracked)
        modules_before = set(sys.modules)
        try:
            entry_point.load()
        except RegistrationConflictError as e:
            _rollback(tracked, before)
            _discard_modules(root_module, modules_before)
            raise PluginConflictError(f'Failed to load {info.describe()}: {e}') from e
        except Exception as e:
            _rollback(tracked, before)
            _discard_modules(root_module, modules_before)
            raise PluginLoadError(f'Failed to load {info.describe()}: {type(e).__name__}: {e}') from e

        overridden = _overridden_names(tracked, before)
        if overridden:
            _rollback(tracked, before)
            _discard_modules(root_module, modules_before)
            raise PluginConflictError(
                f'Failed to load {info.describe()}: it registers names that are already registered: '
                f'{", ".join(overridden)}. Rename them in the plugin.'
            )
        info.registrations = _registered_names(tracked, before)

    summary = '; '.join(f'{kind}: {", ".join(keys)}' for kind, keys in info.registrations.items()) or 'nothing'
    logger.info(f'Loaded {info.describe()}, registered {summary}')
    return info
