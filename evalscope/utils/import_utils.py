# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import importlib
import os
from itertools import chain
from types import ModuleType
from typing import Any, Optional, Union

from .logger import get_logger

logger = get_logger()  # pylint: disable=invalid-name


def check_import(module_name: str, package: Optional[str] = None, raise_error: bool = False) -> bool:
    """Check if a module can be imported.

    Args:
        module_name (str): The name of the module to check.
        package (str, optional): The package to install if the module is not found. Defaults to None.
        raise_error (bool, optional): Whether to raise an error if the module is not found. Defaults to False.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        error_msg = f'`{module_name}` not found.'
        if package:
            error_msg += f' Please run `pip install {package}` to use this feature.'
            logger.warning(error_msg)

        if raise_error:
            raise ImportError(error_msg)
        return False


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f'module {self.__name__} has no attribute {name}')

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        return importlib.import_module('.' + module_name, self.__name__)

    def __reduce__(self):
        return self.__class__, (self._name, self.__file__, self._import_structure)


def is_module_installed(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def get_module_path(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec and spec.origin:
        return os.path.abspath(spec.origin)
    else:
        raise ValueError(f'Cannot find module: {module_name}')
