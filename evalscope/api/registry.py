import copy
from typing import TYPE_CHECKING, Dict, Optional, Type, Union

if TYPE_CHECKING:
    from evalscope.api.benchmark import BenchmarkMeta, DataAdapter
    from evalscope.api.model.model import ModelAPI
    from evalscope.config import TaskConfig
    from evalscope.models import BaseModelAdapter

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
    """Retrieve a registered benchmark by name."""
    # copy to avoid modifying the original metadata
    metadata = copy.deepcopy(BENCHMARK_REGISTRY.get(name))
    if not metadata:
        raise ValueError(f'Benchmark {name} not found, available benchmarks: {list(BENCHMARK_REGISTRY.keys())}')

    # Update metadata with dataset-specific configuration
    if config is not None:
        dataset_config = config.dataset_args.get(name, {})
        metadata._update(dataset_config)
    # Return the data adapter initialized with the benchmark metadata
    data_adapter_cls = metadata.data_adapter
    return data_adapter_cls(benchmark_meta=metadata, task_config=config)


# END: Registry for benchmarks

# BEGIN: Registry for model adapters
# Registry for model adapters, allowing dynamic registration and retrieval of model adapter classes.
MODEL_ADAPTERS: Dict[str, 'BaseModelAdapter'] = {}


def register_model_adapter(name):
    """
    Decorator to register a model adapter with a given name.
    :param name: The name of the model adapter.
    """

    def decorator(adapter):
        if name in MODEL_ADAPTERS:
            raise ValueError(f"Model adapter '{name}' is already registered.")
        MODEL_ADAPTERS[name] = adapter
        return adapter

    return decorator


def get_model_adapter(name) -> Type['BaseModelAdapter']:
    """
    Retrieve a registered model adapter by name.
    :param name: The name of the model adapter.
    :return: The model adapter class or function.
    """
    if name not in MODEL_ADAPTERS:
        raise ValueError(
            f"Model adapter '{name}' is not registered. Available model adapters: {list(MODEL_ADAPTERS.keys())}")
    return MODEL_ADAPTERS[name]


# END: Registry for model adapters

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