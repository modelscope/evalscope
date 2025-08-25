import copy
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union

if TYPE_CHECKING:
    from evalscope.api.benchmark import BenchmarkMeta, DataAdapter
    from evalscope.api.filter import Filter
    from evalscope.api.metric import Aggregator, Metric
    from evalscope.api.model.model import ModelAPI
    from evalscope.config import TaskConfig

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
