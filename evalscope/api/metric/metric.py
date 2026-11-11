import inspect
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, List, Union

from evalscope.utils import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from evalscope.api.evaluator import Target


class MetricUnavailableError(RuntimeError):
    """Invalid scoring inputs or execution failures must remain excluded from recall."""


class Metric(ABC):
    """
    Metric classes operate on a sample level.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Metric class should have state.
        """

    @abstractmethod
    def apply(self, predictions: List[str], references: List[str | List[str]]) -> List[float]:
        pass

    def prepare_reference(self, target: 'Target') -> str:
        """Select a reference representation supported by this metric."""
        return target.single()

    def __call__(self, prediction: str, reference: str | List[str]) -> float:
        """Allows the metric to be called like a function."""
        return self.apply([prediction], [reference])[0]


def _metric_config_key(value: Any) -> Hashable:
    """Freeze configuration containers without conflating their value types."""
    if isinstance(value, dict):
        return type(value), frozenset((_metric_config_key(k), _metric_config_key(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return type(value), tuple(_metric_config_key(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return type(value), frozenset(_metric_config_key(item) for item in value)
    hash(value)
    return type(value), value


class SingletonMetric(Metric):
    """Reuse one initialized scorer per concrete class and constructor configuration.

    Implement initialization in ``_init_once``. Instances remain cached for the process
    lifetime, as model loading may otherwise repeat for every scored sample. Opaque
    unhashable options bypass caching; they are never identified by their repr.
    """

    _instances: dict[Hashable, 'SingletonMetric'] = {}
    _initialization_locks: dict[Hashable, Any] = {}
    _cache_lock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> 'SingletonMetric':
        bound = inspect.signature(cls._init_once).bind(None, *args, **kwargs)
        bound.apply_defaults()
        try:
            key = (cls, _metric_config_key(bound.arguments))
        except TypeError:
            return cls._create_instance(*args, **kwargs)

        # Keep the short cache bookkeeping lock separate from heavyweight model loading.
        with SingletonMetric._cache_lock:
            lock = SingletonMetric._initialization_locks.setdefault(key, threading.RLock())
        with lock:
            if key not in SingletonMetric._instances:
                # Publish only after success. A waiter can retry a failed load with a fresh object.
                SingletonMetric._instances[key] = cls._create_instance(*args, **kwargs)
            return SingletonMetric._instances[key]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # __new__ completes initialization before making an instance available to other callers.
        pass

    @classmethod
    def _create_instance(cls, *args: Any, **kwargs: Any) -> 'SingletonMetric':
        instance = super().__new__(cls)
        logger.info(f'Initializing {cls.__name__}...')
        instance._init_once(*args, **kwargs)
        return instance

    def _init_once(self, *args: Any, **kwargs: Any) -> None:
        pass


class T2IMetric(SingletonMetric):
    """Singleton base class for T2I metrics."""

    @staticmethod
    def _resolve_defaults(device=None, cache_dir=None):
        """Resolve device and cache_dir defaults for T2I metrics."""
        if device is None:
            from evalscope.utils.model_utils import get_device

            device = get_device()
        if cache_dir is None:
            from evalscope.metrics.vision.t2v_metrics.constants import CACHE_DIR

            cache_dir = CACHE_DIR
        return device, cache_dir

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[Union[float, dict]]:
        pass

    def __call__(self, image: str, text: str, **kwargs) -> Union[float, dict]:
        return self.apply([image], [text], **kwargs)[0]
