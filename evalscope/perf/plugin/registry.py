from typing import Any, List, Type, Union


class PluginRegistry:
    _registry = {}

    @classmethod
    def register(cls, name, plugin_cls):
        cls._registry[name] = plugin_cls
        return plugin_cls

    @classmethod
    def get_class(cls, name):
        return cls._registry[name]

    @classmethod
    def all_classes(cls):
        return list(cls._registry.keys())


def register_dataset(name: Union[str, List[str]]):

    def class_decorator(cls: Type):
        if isinstance(name, str):
            DatasetRegistry.register(name, cls)
        elif isinstance(name, list):
            for n in name:
                DatasetRegistry.register(n, cls)
        else:
            raise TypeError('name must be a string or a list of strings')
        return cls

    return class_decorator


def register_api(name: Union[str, List[str]]):

    def class_decorator(cls: Type):
        if isinstance(name, str):
            ApiRegistry.register(name, cls)
        elif isinstance(name, list):
            for n in name:
                ApiRegistry.register(n, cls)
        else:
            raise TypeError('name must be a string or a list of strings')
        return cls

    return class_decorator


class DatasetRegistry(PluginRegistry):
    """Registry for dataset plugins."""
    _registry = {}

    @classmethod
    def get_class(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Dataset plugin '{name}' is not registered.")
        return cls._registry[name]


class ApiRegistry(PluginRegistry):
    """Registry for API plugins."""
    _registry = {}

    @classmethod
    def get_class(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"API plugin '{name}' is not registered.")
        return cls._registry[name]
