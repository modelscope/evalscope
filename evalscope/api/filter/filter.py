from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, List, Union


class Filter(ABC):
    """
    Filter classes operate on a sample level.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    @abstractmethod
    def apply(self, instance: List[str]) -> List[str]:

        return instance

    def __call__(self, instance: str) -> str:
        """
        Allows the filter to be called like a function.
        """
        return self.apply([instance])[0]


@dataclass
class FilterEnsemble:
    """
    FilterEnsemble creates a pipeline applying multiple filters.
    Its intended usage is to stack multiple post-processing steps in order.
    """

    name: str
    filters: List[Callable[[], Filter]]

    def apply(self, instance: List[str]) -> List[str]:

        for f in self.filters:
            # apply filters in sequence
            instance = f().apply(instance)

        return instance

    def __call__(self, instance: str) -> str:
        """
        Allows the filter ensemble to be called like a function.
        """
        return self.apply([instance])[0]
