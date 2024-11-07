from typing import Any


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


def register_dataset(name: str):

    def class_decorator(cls):
        DatasetRegistry.register(name, cls)
        return cls

    return class_decorator


def register_api(name: str):

    def class_decorator(cls):
        ApiRegistry.register(name, cls)
        return cls

    return class_decorator


DatasetRegistry = PluginRegistry()
ApiRegistry = PluginRegistry()
