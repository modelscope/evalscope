# Copyright (c) Alibaba, Inc. and its affiliates.

from llmuses.utils.logger import get_logger

TYPE_NAME = 'type'
DEFAULT_GROUP = 'default'
logger = get_logger()
AST_INDEX = None


class Registry(object):

    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def __repr__(self):
        format_str = self.__class__.__name__ + f' ({self._name})\n'
        for obj_name, obj in self._registry.items():
            format_str += f'objects={obj_name}\n'

        return format_str

    @property
    def name(self):
        return self._name

    @property
    def registry(self):
        return self._registry

    def list(self):
        """ logging the list of objects in current registry
        """
        logger.info(f'objects={list(self._registry.keys())}')

    def get(self, obj_name: str):
        return self._registry.get(obj_name, None)

    def register(self, obj_name: str, force=False):
        """ Register a object to current registry

        Args:
            obj_name (str): object name
            force (bool, optional): whether to overwrite the existing object
                with the same name. Defaults to False.
        """
        if obj_name in self._registry and not force:
            raise KeyError(f'{obj_name} is already registered in {self.name}')

        def _register(obj_cls):
            self._registry[obj_name] = obj_cls
            return obj_cls

        return _register


def get_registered_obj(registry: Registry, obj_name: str):
    """ Get registered object from registry

    Args:
        registry (str): registry instance
        obj_name (str): object name

    Returns:
        Registry: registry object
    """
    return registry.get(obj_name=obj_name)
