import copy
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Type, Union

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

# BEGIN: Registry for benchmarks
# Registry for benchmarks, allowing dynamic registration and retrieval of benchmark metadata and data adapters.
BENCHMARK_REGISTRY: Dict[str, 'BenchmarkMeta'] = {}


def register_benchmark(metadata: 'BenchmarkMeta'):
    """Register a benchmark with its metadata."""

    def register_wrapper(data_adapter: Type['DataAdapter']):
        if metadata.name in BENCHMARK_REGISTRY:
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
        dataset_args (Optional[dict]): The dataset-specific arguments.

    """
    # copy to avoid modifying the original metadata
    metadata = copy.deepcopy(BENCHMARK_REGISTRY.get(name))
    if not metadata:
        raise ValueError(f'Benchmark {name} not found, available benchmarks: {list(sorted(BENCHMARK_REGISTRY.keys()))}')

    # Update metadata with dataset-specific configuration
    if config is not None:
        metadata._update(config.dataset_args.get(name, {}))
    # Return the data adapter initialized with the benchmark metadata
    data_adapter_cls = metadata.data_adapter
    return data_adapter_cls(benchmark_meta=metadata, task_config=config)


# END: Registry for benchmarks

# BEGIN: Registry for model APIs
# Registry for model APIs, allowing dynamic registration and retrieval of model API classes.
MODEL_APIS: Dict[str, Type['ModelAPI']] = {}


def register_model_api(name: str):
    """
    Decorator to register a model API class with a given name.

    :param name: The name of the model API.
    """

    def decorator(api_class: Type['ModelAPI']):
        if name in MODEL_APIS:
            raise ValueError(f"Model API '{name}' is already registered.")
        MODEL_APIS[name] = api_class
        return api_class

    return decorator


def get_model_api(name: str) -> Type['ModelAPI']:
    """
    Retrieve a registered model API class by name.

    :param name: The name of the model API.
    :return: The model API class.
    """
    if name not in MODEL_APIS:
        raise ValueError(f"Model API '{name}' is not registered. Available model APIs: {list(MODEL_APIS.keys())}")

    wrapped = MODEL_APIS[name]
    if not isinstance(wrapped, type):
        return wrapped()
    else:
        return wrapped


# END: Registry for model APIs

# BEGIN: Registry for metrics
METRIC_REGISTRY: Dict[str, Type['Metric']] = {}


def register_metric(name: str):

    def decorate(fn):
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric named '{name}' conflicts with existing registered metric!")

        METRIC_REGISTRY[name] = fn
        return fn

    return decorate


def get_metric(name: str) -> Type['Metric']:
    if name in METRIC_REGISTRY:
        return METRIC_REGISTRY[name]
    else:
        raise ValueError(
            f"Metric '{name}' not found in the registry. Available metrics: {list(METRIC_REGISTRY.keys())}"
        )


# END: Registry for metrics

# BEGIN: Registry for filters

FILTER_REGISTRY: Dict[str, Type['Filter']] = {}


def register_filter(name):

    def decorate(cls):
        if name in FILTER_REGISTRY:
            raise ValueError(f'Registering filter `{name}` that is already in Registry {FILTER_REGISTRY}')
        FILTER_REGISTRY[name] = cls
        return cls

    return decorate


def get_filter(filter_name: str) -> Type['Filter']:
    if filter_name not in FILTER_REGISTRY:
        raise KeyError(
            f"Filter '{filter_name}' not found in the registry. Available filters: {list(FILTER_REGISTRY.keys())}"
        )
    return FILTER_REGISTRY[filter_name]


# END: Registry for filters

# BEGIN: Registry for aggregation functions
AGGREGATION_REGISTRY: Dict[str, Type['Aggregator']] = {}


def register_aggregation(name: str):
    """
    Decorator to register an aggregation function with a given name.

    :param name: The name of the aggregation function.
    """

    def decorator(aggregation_fn: 'Aggregator'):
        if name in AGGREGATION_REGISTRY:
            raise ValueError(f"Aggregation function '{name}' is already registered.")
        AGGREGATION_REGISTRY[name] = aggregation_fn
        return aggregation_fn

    return decorator


def get_aggregation(name: str) -> Type['Aggregator']:
    """
    Retrieve a registered aggregation function by name.

    :param name: The name of the aggregation function.
    :return: The aggregation function.
    """
    if name not in AGGREGATION_REGISTRY:
        raise ValueError(
            f"Aggregation function '{name}' is not registered. "
            f'Available aggregations: {list(AGGREGATION_REGISTRY.keys())}'
        )
    return AGGREGATION_REGISTRY[name]


# END: Registry for aggregation functions

# BEGIN: Registry for evaluators
# Registry for evaluators, allowing benchmarks to register a custom Evaluator class.
# Concrete evaluator classes self-register via @register_evaluator('name').
EVALUATOR_REGISTRY: Dict[str, Type['Evaluator']] = {}


def register_evaluator(name: str):
    """
    Decorator to register an Evaluator class under a given name.

    Usage::

        @register_evaluator('default')
        class DefaultEvaluator(Evaluator):
            ...

    :param name: Registry key (e.g. ``'default'`` or a benchmark name).
    """

    def decorator(cls: Type['Evaluator']) -> Type['Evaluator']:
        if name in EVALUATOR_REGISTRY:
            raise ValueError(f"Evaluator '{name}' is already registered.")
        EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorator


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

    Args:
        benchmark: The data adapter for the benchmark to evaluate.
        model: The model to be evaluated.
        outputs: Output directory structure for saving results.
        task_config: The task configuration.

    Returns:
        A fully initialised :class:`Evaluator` instance.
    """
    evaluator_cls = EVALUATOR_REGISTRY.get(benchmark.name) or EVALUATOR_REGISTRY['default']
    return evaluator_cls(
        benchmark=benchmark,
        model=model,
        outputs=outputs,
        task_config=task_config,
    )


# END: Registry for evaluators

# BEGIN: Registry for agent strategies, environments and tools
# Pluggable pieces that compose the Agent Loop.  Concrete strategy /
# environment / tool classes self-register via the decorators below.
STRATEGY_REGISTRY: Dict[str, Type['AgentStrategy']] = {}
ENVIRONMENT_REGISTRY: Dict[str, Type['AgentEnvironment']] = {}
AGENT_TOOL_REGISTRY: Dict[str, 'ToolHandler'] = {}
AGENT_TOOL_INFO_REGISTRY: Dict[str, 'ToolInfo'] = {}
"""Maps tool name → :class:`ToolInfo` schema.  Populated by :func:`register_agent_tool`
when an ``info`` kwarg is supplied."""


def register_strategy(name: str) -> Callable[[Type['AgentStrategy']], Type['AgentStrategy']]:
    """Register an :class:`AgentStrategy` implementation under ``name``."""

    def decorator(cls: Type['AgentStrategy']) -> Type['AgentStrategy']:
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"Agent strategy '{name}' is already registered.")
        STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


def get_strategy(name: str) -> Type['AgentStrategy']:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Agent strategy '{name}' is not registered. "
            f'Available: {sorted(STRATEGY_REGISTRY.keys())}'
        )
    return STRATEGY_REGISTRY[name]


def list_strategies() -> List[str]:
    return sorted(STRATEGY_REGISTRY.keys())


def register_environment(name: str) -> Callable[[Type['AgentEnvironment']], Type['AgentEnvironment']]:
    """Register an :class:`AgentEnvironment` implementation under ``name``."""

    def decorator(cls: Type['AgentEnvironment']) -> Type['AgentEnvironment']:
        if name in ENVIRONMENT_REGISTRY:
            raise ValueError(f"Agent environment '{name}' is already registered.")
        ENVIRONMENT_REGISTRY[name] = cls
        return cls

    return decorator


def get_environment(name: str) -> Type['AgentEnvironment']:
    if name not in ENVIRONMENT_REGISTRY:
        raise ValueError(
            f"Agent environment '{name}' is not registered. "
            f'Available: {sorted(ENVIRONMENT_REGISTRY.keys())}'
        )
    return ENVIRONMENT_REGISTRY[name]


def list_environments() -> List[str]:
    return sorted(ENVIRONMENT_REGISTRY.keys())


# BEGIN: Registry for external-agent runners
RUNNER_REGISTRY: Dict[str, Type['AgentRunner']] = {}


def register_runner(name: str) -> Callable[[Type['AgentRunner']], Type['AgentRunner']]:
    """Register an :class:`AgentRunner` implementation under ``name``.

    Mirrors :func:`register_environment`.  The ``name`` is what
    :class:`ExternalAgentConfig.framework` resolves through
    :func:`get_runner` at TaskConfig validation time.
    """

    def decorator(cls: Type['AgentRunner']) -> Type['AgentRunner']:
        if name in RUNNER_REGISTRY:
            raise ValueError(f"Agent runner '{name}' is already registered.")
        RUNNER_REGISTRY[name] = cls
        cls.framework = name
        return cls

    return decorator


def get_runner(name: str) -> Type['AgentRunner']:
    if name not in RUNNER_REGISTRY:
        raise ValueError(f"Agent runner '{name}' is not registered. " + f'Available: {sorted(RUNNER_REGISTRY.keys())}')
    return RUNNER_REGISTRY[name]


def list_runners() -> List[str]:
    return sorted(RUNNER_REGISTRY.keys())


def register_agent_tool(
    name: str,
    info: Optional['ToolInfo'] = None,
) -> Callable[['ToolHandler'], 'ToolHandler']:
    """Register an async tool handler under ``name``.

    The decorated callable must match :data:`ToolHandler`:
    ``async def run(call: ToolCall, env: Optional[AgentEnvironment]) -> str``.

    Args:
        name:  Registry key (also used as the tool function name exposed to the model).
        info:  Optional :class:`~evalscope.api.tool.ToolInfo` schema.  When provided, it
               is stored in :data:`AGENT_TOOL_INFO_REGISTRY` so that
               :func:`resolve_tool_infos` can surface it to ``model.generate``.
    """

    def decorator(fn: 'ToolHandler') -> 'ToolHandler':
        if name in AGENT_TOOL_REGISTRY:
            raise ValueError(f"Agent tool '{name}' is already registered.")
        AGENT_TOOL_REGISTRY[name] = fn
        if info is not None:
            AGENT_TOOL_INFO_REGISTRY[name] = info
        return fn

    return decorator


def get_agent_tool(name: str) -> 'ToolHandler':
    if name not in AGENT_TOOL_REGISTRY:
        raise ValueError(f"Agent tool '{name}' is not registered. "
                         f'Available: {sorted(AGENT_TOOL_REGISTRY.keys())}')
    return AGENT_TOOL_REGISTRY[name]


def list_agent_tools() -> List[str]:
    return sorted(AGENT_TOOL_REGISTRY.keys())


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


# END: Registry for agent strategies, environments and tools
