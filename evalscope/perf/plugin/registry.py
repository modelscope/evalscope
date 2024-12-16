from typing import Any, List, Type, Union


class PluginRegistry:

    def __init__(self):
        self._registry = {}

    def register(self, name, cls):
        self._registry[name] = cls
        return cls

    def get_class(self, name):
        return self._registry[name]

    def all_classes(self):
        return list(self._registry.keys())

    def __call__(self, name: str) -> Any:
        return self.get_class(name)


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


DatasetRegistry = PluginRegistry()
ApiRegistry = PluginRegistry()
