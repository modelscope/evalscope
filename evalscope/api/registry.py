import copy
import difflib
from contextlib import contextmanager
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Type, TypeVar, Union

if TYPE_CHECKING:
    from evalscope.agent.external.runners.base import AgentRunner
    from evalscope.api.agent import AgentEnvironment, AgentStrategy, ToolHandler
    from evalscope.api.benchmark import BenchmarkMeta, DataAdapter
    from evalscope.api.evaluator import Evaluator
    from evalscope.api.filter import Filter
    from evalscope.api.metric import Aggregator, Metric
    from evalscope.api.model.model import ModelAPI
    from evalscope.api.tool import ToolInfo
    from evalscope.config import TaskConfig
    from evalscope.utils.io_utils import OutputsStructure

# BEGIN: Registry base
T = TypeVar('T')
_MISSING = object()


class Registry(Dict[str, T]):
    """Generic name → object registry with alias support.

    Subclasses ``dict`` so existing call sites that use ``.keys()``, ``.values()``,
    ``.items()``, ``in``, ``[]`` and ``.pop()`` keep working unchanged.
    """

    def __init__(
        self,
        kind: str,
        *,
        on_register: Optional[Callable[[Any, List[str]], None]] = None,
    ) -> None:
        super().__init__()
        self.kind = kind
        self._on_register = on_register

    def register(self, name: Union[str, List[str]]) -> Callable[[T], T]:
        """Decorator that registers a value under one or more names.

        Passing a list registers the value under every name, with the first
        entry treated as the canonical / primary name.
        """
        names = [name] if isinstance(name, str) else list(name)
        if not names:
            raise ValueError(f'{self.kind} registration requires at least one name.')

        def decorator(obj: T) -> T:
            for n in names:
                if self.is_registered(n):
                    raise ValueError(f"{self.kind} '{n}' is already registered.")
            if self._on_register is not None:
                self._on_register(obj, names)
            for n in names:
                self[n] = obj
            return obj

        return decorator

    def is_registered(self, name: str, registering_module: Optional[str] = None) -> bool:
        """Whether ``name`` is already present, without resolving anything.

        ``registering_module`` is accepted for the lazy benchmark registry, which
        reserves indexed built-in names without loading them. Other registries only
        look at materialized entries.
        """
        del registering_module
        return dict.__contains__(self, name)

    def is_materialized(self, name: str) -> bool:
        """Whether ``name`` is in the backing dictionary without resolving it."""
        return dict.__contains__(self, name)

    def _suggest(self, name: str, n: int = 2) -> str:
        """Return a hint string with the closest registered names by edit distance."""
        candidates = difflib.get_close_matches(name, self.keys(), n=n, cutoff=0.4)
        if candidates:
            return f'Did you mean: {", ".join(repr(c) for c in candidates)}?'
        # Fallback: show up to 10 entries if nothing is close enough
        keys = sorted(self.keys())
        if len(keys) > 10:
            return f'Available ({len(keys)} total, showing first 10): {keys[:10]}'
        return f'Available: {keys}'

    def lookup(self, name: str) -> T:
        """Get the value registered under ``name`` or raise with suggestions."""
        if name not in self:
            raise ValueError(f"{self.kind} '{name}' is not registered. {self._suggest(name)}")
        return self[name]

    def list_keys(self) -> List[str]:
        return sorted(self.keys())


class LazyRegistry(Registry[T]):
    """Registry that materializes entries on demand.

    Discovery (which names exist) is separated from loading (executing the module
    that registers a name), so importing the owning package does not have to
    execute every plugin module. The owning package installs the resolvers via
    :meth:`set_resolvers`; until it does, this behaves exactly like
    :class:`Registry`.

    Reads that can be satisfied by a single name resolve that name only; reads that
    need the complete catalog (``keys``/``values``/``items``/iteration/``len``)
    load everything, so batch consumers keep seeing the full set. Resolution holds
    an ``RLock`` across module import: another thread waits for the first importer
    instead of observing a temporarily empty registry and treating a valid name as
    unknown.
    """

    def __init__(
        self,
        kind: str,
        *,
        on_register: Optional[Callable[[Any, List[str]], None]] = None,
    ) -> None:
        super().__init__(kind, on_register=on_register)
        self._resolve_one: Optional[Callable[[str], bool]] = None
        self._resolve_all: Optional[Callable[[], None]] = None
        self._indexed_module: Optional[Callable[[str], Optional[str]]] = None
        self._lock = RLock()
        self._allow_indexed_registrations = 0

    def set_resolvers(
        self,
        resolve_one: Callable[[str], bool],
        resolve_all: Callable[[], None],
        indexed_module: Callable[[str], Optional[str]],
    ) -> None:
        """Install the loaders and the generated name-to-module lookup.

        Args:
            resolve_one: Load whatever registers a single name; return False when the
                name is unknown or stale in the caller's index, so the full load is used.
            resolve_all: Load every plugin module.
            indexed_module: Return the module expected to register a name without
                importing it. This reserves shipped names from custom overrides.
        """
        with self._lock:
            self._resolve_one = resolve_one
            self._resolve_all = resolve_all
            self._indexed_module = indexed_module

    @contextmanager
    def allow_indexed_registrations(self) -> Iterator[None]:
        """Allow a full discovery scan to repair a stale generated index."""
        with self._lock:
            self._allow_indexed_registrations += 1
            try:
                yield
            finally:
                self._allow_indexed_registrations -= 1

    def is_registered(self, name: str, registering_module: Optional[str] = None) -> bool:
        """Check materialized entries and reserve indexed built-in names.

        An adapter is allowed to register a shipped name only when its own module is
        the module recorded in the index. A full scan temporarily bypasses the index
        reservation so it can recover from a stale mapping.
        """
        with self._lock:
            if dict.__contains__(self, name):
                return True
            if self._allow_indexed_registrations or self._indexed_module is None:
                return False
            indexed_module = self._indexed_module(name)
            return indexed_module is not None and indexed_module != registering_module

    def _materialize(self, name: str) -> None:
        """Resolve one name, falling back to a full load when it is not indexed."""
        with self._lock:
            if self._resolve_one is None or dict.__contains__(self, name):
                return
            if not self._resolve_one(name) and self._resolve_all is not None:
                self._resolve_all()

    def _materialize_all(self) -> None:
        """Resolve the complete catalog."""
        with self._lock:
            if self._resolve_all is not None:
                self._resolve_all()

    def __contains__(self, name: object) -> bool:
        if isinstance(name, str):
            self._materialize(name)
        with self._lock:
            return dict.__contains__(self, name)

    def __getitem__(self, name: str) -> T:
        self._materialize(name)
        with self._lock:
            return dict.__getitem__(self, name)

    def get(self, name: str, default: Any = None) -> Any:
        self._materialize(name)
        with self._lock:
            return dict.get(self, name, default)

    def keys(self):
        self._materialize_all()
        with self._lock:
            return dict.keys(self)

    def values(self):
        self._materialize_all()
        with self._lock:
            return dict.values(self)

    def items(self):
        self._materialize_all()
        with self._lock:
            return dict.items(self)

    def __iter__(self):
        self._materialize_all()
        with self._lock:
            return dict.__iter__(self)

    def __len__(self) -> int:
        self._materialize_all()
        with self._lock:
            return dict.__len__(self)

    def pop(self, name: str, default: Any = _MISSING) -> Any:
        """Remove and return one entry, materializing it first.

        ``Registry`` documents ``pop()`` as part of its dict-compatible public
        surface. Without this override, popping a valid lazy benchmark before any
        lookup would incorrectly raise ``KeyError`` just because its adapter module
        had not run yet.
        """
        self._materialize(name)
        with self._lock:
            if default is _MISSING:
                return dict.pop(self, name)
            return dict.pop(self, name, default)

    def setdefault(self, name: str, default: Optional[T] = None) -> Optional[T]:
        """Materialize ``name`` before applying normal dict ``setdefault`` semantics."""
        self._materialize(name)
        with self._lock:
            return dict.setdefault(self, name, default)


# END: Registry base

# BEGIN: Registry for benchmarks
# Stores BenchmarkMeta (not the adapter class) because the adapter is attached
# to the metadata at registration time.
# Lazy: ``evalscope.benchmarks`` installs the resolvers that import adapter modules
# on demand, so a single-benchmark run does not execute every adapter module.
BENCHMARK_REGISTRY: LazyRegistry['BenchmarkMeta'] = LazyRegistry('Benchmark')


def register_benchmark(metadata: 'BenchmarkMeta'):
    """Register a benchmark with its metadata."""

    def register_wrapper(data_adapter: Type['DataAdapter']):
        if BENCHMARK_REGISTRY.is_registered(metadata.name, data_adapter.__module__):
            raise ValueError(f'Benchmark {metadata.name} already registered')
        metadata.data_adapter = data_adapter
        BENCHMARK_REGISTRY[metadata.name] = metadata
        return data_adapter

    return register_wrapper


def get_benchmark(name: str, config: Optional['TaskConfig'] = None) -> 'DataAdapter':
    """
    Retrieve a registered benchmark by name.

    Args:
        name (str): The name of the benchmark.
        config (Optional['TaskConfig']): The task configuration.
    """
    # copy to avoid modifying the original metadata
    metadata = copy.deepcopy(BENCHMARK_REGISTRY.get(name))
    if not metadata:
        raise ValueError(f"Benchmark '{name}' not found. {BENCHMARK_REGISTRY._suggest(name)}")

    # Update metadata with dataset-specific configuration
    if config is not None:
        metadata._update(config.dataset_args.get(name, {}))
    # Construction must be side-effect free: callers also use this helper for metadata and docs.
    data_adapter_cls = metadata.data_adapter
    return data_adapter_cls(benchmark_meta=metadata, task_config=config)


# END: Registry for benchmarks

# BEGIN: Registry for model APIs
MODEL_APIS: Registry[Type['ModelAPI']] = Registry('Model API')


def register_model_api(name: Union[str, List[str]]):
    """Decorator to register a model API class under one or more names."""
    return MODEL_APIS.register(name)


def get_model_api(name: str) -> Type['ModelAPI']:
    """Retrieve a registered model API class by name."""
    wrapped = MODEL_APIS.lookup(name)
    if not isinstance(wrapped, type):
        return wrapped()
    return wrapped


# END: Registry for model APIs

# BEGIN: Registry for metrics
METRIC_REGISTRY: Registry[Type['Metric']] = Registry('Metric')


def register_metric(name: Union[str, List[str]]):
    return METRIC_REGISTRY.register(name)


def get_metric(name: str) -> Type['Metric']:
    return METRIC_REGISTRY.lookup(name)


# END: Registry for metrics

# BEGIN: Registry for filters
FILTER_REGISTRY: Registry[Type['Filter']] = Registry('Filter')


def register_filter(name: Union[str, List[str]]):
    return FILTER_REGISTRY.register(name)


def get_filter(filter_name: str) -> Type['Filter']:
    return FILTER_REGISTRY.lookup(filter_name)


# END: Registry for filters

# BEGIN: Registry for aggregation functions
AGGREGATION_REGISTRY: Registry[Type['Aggregator']] = Registry('Aggregation function')


def register_aggregation(name: Union[str, List[str]]):
    """Decorator to register an aggregation function under one or more names."""
    return AGGREGATION_REGISTRY.register(name)


def get_aggregation(name: str) -> Type['Aggregator']:
    """Retrieve a registered aggregation function by name."""
    return AGGREGATION_REGISTRY.lookup(name)


# END: Registry for aggregation functions

# BEGIN: Registry for evaluators
# Concrete evaluator classes self-register via @register_evaluator('name').
EVALUATOR_REGISTRY: Registry[Type['Evaluator']] = Registry('Evaluator')


def register_evaluator(name: Union[str, List[str]]):
    """Decorator to register an Evaluator class under one or more names."""
    return EVALUATOR_REGISTRY.register(name)


def create_evaluator(
    benchmark: 'DataAdapter',
    model,
    outputs: 'OutputsStructure',
    task_config: 'TaskConfig',
) -> 'Evaluator':
    """
    Instantiate the appropriate :class:`Evaluator` for the given benchmark.

    Looks up ``benchmark.name`` in :data:`EVALUATOR_REGISTRY`; if not found,
    falls back to the ``'default'`` entry (i.e. :class:`DefaultEvaluator`).
    """
    evaluator_cls = EVALUATOR_REGISTRY.get(benchmark.name) or EVALUATOR_REGISTRY['default']
    return evaluator_cls(
        benchmark=benchmark,
        model=model,
        outputs=outputs,
        task_config=task_config,
    )


# END: Registry for evaluators

# BEGIN: Registry for agent strategies, environments, runners and tools
STRATEGY_REGISTRY: Registry[Type['AgentStrategy']] = Registry('Agent strategy')
ENVIRONMENT_REGISTRY: Registry[Type['AgentEnvironment']] = Registry('Agent environment')
AGENT_TOOL_REGISTRY: Registry['ToolHandler'] = Registry('Agent tool')
AGENT_TOOL_INFO_REGISTRY: Dict[str, 'ToolInfo'] = {}
"""Maps tool name → :class:`ToolInfo` schema.  Populated by :func:`register_agent_tool`
when an ``info`` kwarg is supplied."""


def register_strategy(name: Union[str, List[str]]) -> Callable[[Type['AgentStrategy']], Type['AgentStrategy']]:
    """Register an :class:`AgentStrategy` implementation under one or more names."""
    return STRATEGY_REGISTRY.register(name)


def get_strategy(name: str) -> Type['AgentStrategy']:
    return STRATEGY_REGISTRY.lookup(name)


def list_strategies() -> List[str]:
    return STRATEGY_REGISTRY.list_keys()


def register_environment(name: Union[str, List[str]]) -> Callable[[Type['AgentEnvironment']], Type['AgentEnvironment']]:
    """Register an :class:`AgentEnvironment` implementation under one or more names."""
    return ENVIRONMENT_REGISTRY.register(name)


def get_environment(name: str) -> Type['AgentEnvironment']:
    return ENVIRONMENT_REGISTRY.lookup(name)


def list_environments() -> List[str]:
    return ENVIRONMENT_REGISTRY.list_keys()


# Runner registry: the first registered name becomes the canonical framework
# label on the class (consumed by bridge/server.py and the external adapter).
def _set_runner_framework(cls: Type['AgentRunner'], names: List[str]) -> None:
    cls.framework = names[0]


RUNNER_REGISTRY: Registry[Type['AgentRunner']] = Registry('Agent runner', on_register=_set_runner_framework)


def register_runner(name: Union[str, List[str]]) -> Callable[[Type['AgentRunner']], Type['AgentRunner']]:
    """Register an :class:`AgentRunner` implementation under one or more names.

    Mirrors :func:`register_environment`. The first ``name`` becomes the
    canonical ``cls.framework`` value that :class:`ExternalAgentConfig.framework`
    resolves through :func:`get_runner` at TaskConfig validation time.
    """
    return RUNNER_REGISTRY.register(name)


def get_runner(name: str) -> Type['AgentRunner']:
    return RUNNER_REGISTRY.lookup(name)


def list_runners() -> List[str]:
    return RUNNER_REGISTRY.list_keys()


def register_agent_tool(
    name: Union[str, List[str]],
    info: Optional['ToolInfo'] = None,
) -> Callable[['ToolHandler'], 'ToolHandler']:
    """Register an async tool handler under one or more names.

    The decorated callable must match :data:`ToolHandler`:
    ``async def run(call: ToolCall, env: Optional[AgentEnvironment]) -> ToolObservation``.

    Args:
        name:  Registry key(s) (also used as the tool function name exposed to the model).
        info:  Optional :class:`~evalscope.api.tool.ToolInfo` schema.  When provided, it
               is stored in :data:`AGENT_TOOL_INFO_REGISTRY` under every name so that
               :func:`resolve_tool_infos` can surface it to ``model.generate``.
    """
    names = [name] if isinstance(name, str) else list(name)

    def decorator(fn: 'ToolHandler') -> 'ToolHandler':
        AGENT_TOOL_REGISTRY.register(names)(fn)
        if info is not None:
            for n in names:
                AGENT_TOOL_INFO_REGISTRY[n] = info
        return fn

    return decorator


def get_agent_tool(name: str) -> 'ToolHandler':
    return AGENT_TOOL_REGISTRY.lookup(name)


def list_agent_tools() -> List[str]:
    return AGENT_TOOL_REGISTRY.list_keys()


def resolve_tools(names: Optional[List[str]]) -> Dict[str, 'ToolHandler']:
    """Look up multiple tool handlers by name.  ``None`` / empty → ``{}``."""
    if not names:
        return {}
    return {name: get_agent_tool(name) for name in names}


def get_agent_tool_info(name: str) -> Optional['ToolInfo']:
    """Return the :class:`ToolInfo` schema for a registered tool, or ``None``."""
    return AGENT_TOOL_INFO_REGISTRY.get(name)


def resolve_tool_infos(names: Optional[List[str]]) -> List['ToolInfo']:
    """Return :class:`ToolInfo` schemas for the named tools that have one.

    Tools registered without an ``info`` argument are silently skipped.
    ``None`` / empty → ``[]``.
    """
    if not names:
        return []
    return [AGENT_TOOL_INFO_REGISTRY[n] for n in names if n in AGENT_TOOL_INFO_REGISTRY]


# END: Registry for agent strategies, environments, runners and tools
