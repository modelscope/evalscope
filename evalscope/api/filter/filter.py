from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Union

from evalscope.api.registry import get_filter


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
            instance = f.apply(instance)

        return instance

    def __call__(self, instance: str) -> str:
        """
        Allows the filter ensemble to be called like a function.
        """
        return self.apply([instance])[0]


def build_filter_ensemble(name: str = 'default', filters: Dict[str, Any] = {}) -> FilterEnsemble:
    """
    Create a filtering pipeline.
    """
    filter_funcs = []
    for filter_name, filter_args in filters.items():
        filter_cls = get_filter(filter_name)
        if isinstance(filter_args, list):
            filter_function = filter_cls(*filter_args)
        elif isinstance(filter_args, dict):
            filter_function = filter_cls(**filter_args)
        else:
            # Assume single value for simple filters
            filter_function = filter_cls(filter_args)
        # add the filter as a pipeline step
        filter_funcs.append(filter_function)

    return FilterEnsemble(name=name, filters=filter_funcs)
