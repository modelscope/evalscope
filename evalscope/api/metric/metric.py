from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Union


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
