import re
from typing import Any, Callable, Dict


class Filter:
    """
    A base Filter class that implements the registry pattern
    """
    _registry: Dict[str, Callable[[str, Any], str]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a new filter function
        """

        def decorator(func: Callable[[str, Any], str]) -> Callable[[str, Any], str]:
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def get_filter(cls, name: str) -> Callable:
        """
        Get a registered filter by name
        """
        return cls._registry.get(name)

    @classmethod
    def apply(cls, name: str, value: str, *args, **kwargs) -> str:
        """
        Apply a registered filter to a value
        """
        filter_func = cls.get_filter(name)
        if filter_func is None:
            raise ValueError(f'Filter {name} not found')
        return filter_func(value, *args, **kwargs)


@Filter.register('remove_until')
def remove_until(value: str, marker: str) -> str:
    """
    Remove everything before the last occurrence of marker
    """
    if marker not in value:
        return value
    return value[value.rindex(marker) + len(marker):]


@Filter.register('extract')
def extract(value: str, pattern: str) -> str:
    """
    Extract content from string using regex pattern
    """
    match = re.search(pattern, value)
    if match:
        return match.group(0)
    return ''
