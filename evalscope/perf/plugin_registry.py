
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

dataset_registry = PluginRegistry()
api_registry = PluginRegistry()

def register_dataset(name: str):
    def class_decorator(cls):
        dataset_registry.register(name, cls)
        return cls
    return class_decorator

def register_api(name: str):
    def class_decorator(cls):
        api_registry.register(name, cls)
        return cls
    return class_decorator