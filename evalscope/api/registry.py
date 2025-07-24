from typing import TYPE_CHECKING, Dict, Union

if TYPE_CHECKING:
    from evalscope.api.benchmark import BenchmarkMeta, DataAdapter
    from evalscope.config import TaskConfig

# Registry for benchmarks, allowing dynamic registration and retrieval of benchmark metadata and data adapters.
BENCHMARK_REGISTRY: Dict[str, 'BenchmarkMeta'] = {}


def register_benchmark(metadata: 'BenchmarkMeta'):
    """Register a benchmark with its metadata."""

    def register_wrapper(data_adapter: 'DataAdapter'):
        if metadata.name in BENCHMARK_REGISTRY:
            raise ValueError(f'Benchmark {metadata.name} already registered')
        metadata.data_adapter = data_adapter
        BENCHMARK_REGISTRY[metadata.name] = metadata
        return data_adapter

    return register_wrapper


def get_benchmark(name: str, config: 'TaskConfig') -> 'DataAdapter':
    """Retrieve a registered benchmark by name."""
    metadata = BENCHMARK_REGISTRY.get(name)
    if not metadata:
        raise ValueError(f'Benchmark {name} not found')

    # Update metadata with dataset-specific configuration
    dataset_config = config.dataset_args.get(name, {})
    metadata._update(dataset_config)
    # Return the data adapter initialized with the benchmark metadata
    data_adapter_cls = metadata.data_adapter
    return data_adapter_cls(benchmark_meta=metadata)
