from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Union

from evalscope.utils import get_logger
from evalscope.utils.function_utils import thread_safe

logger = get_logger()


class Metric(ABC):
    """
    Metric classes operate on a sample level.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Metric class should have state.
        """

    @abstractmethod
    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        pass

    def __call__(self, prediction: str, reference: str) -> float:
        """
        Allows the metric to be called like a function.
        """
        return self.apply([prediction], [reference])[0]


class SingletonMetric(Metric):
    """Singleton base class for metrics."""
    _instance = None

    @thread_safe
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        cls = self.__class__
        if hasattr(self, '_init_done'):
            return
        logger.info(f'Initializing {cls.__name__}...')
        self._init_once(*args, **kwargs)
        self._init_done = True

    def _init_once(self, *args, **kwargs):
        pass


class T2IMetric(SingletonMetric):
    """Singleton base class for T2I metrics."""

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[Union[float, dict]]:
        pass

    def __call__(self, image: str, text: str, **kwargs) -> Union[float, dict]:
        return self.apply([image], [text], **kwargs)[0]
