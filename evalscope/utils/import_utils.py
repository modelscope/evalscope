# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import importlib
import os
from itertools import chain
from types import ModuleType
from typing import Any, Optional, Union

from evalscope.constants import IS_BUILD_DOC
from .logger import get_logger

logger = get_logger()  # pylint: disable=invalid-name


def check_import(
    module_name: Union[str, list[str]],
    package: Optional[Union[str, list[str]]] = None,
    raise_warning: bool = True,
    raise_error: bool = False,
    feature_name: Optional[str] = 'this feature',
) -> bool:
    """Check if a module or list of modules can be imported.

    Args:
        module_name (Union[str, list[str]]): The name(s) of the module(s) to check.
        package (Union[str, list[str]], optional): The package(s) to install if the module(s) are not found.
            Defaults to None.
        raise_error (bool, optional): Whether to raise an error if any module is not found. Defaults to False.
        raise_warning (bool, optional): Whether to log a warning if any module is not found. Defaults to True.
        feature_name (str, optional): The feature name that requires the module(s). Used in the warning/error message.
            Defaults to 'this feature'.

    Returns:
        bool: True if all modules can be imported, False otherwise.
    """
    # Convert single strings to lists for uniform processing
    if isinstance(module_name, str):
        module_names = [module_name]
    else:
        module_names = module_name

    if package is None:
        packages = [None] * len(module_names)
    elif isinstance(package, str):
        packages = [package] * len(module_names)
    else:
        packages = package
        # Ensure packages list has same length as module_names
        if len(packages) < len(module_names):
            packages.extend([None] * (len(module_names) - len(packages)))

    missing_modules = []
    missing_packages = []

    for i, mod_name in enumerate(module_names):
        try:
            importlib.import_module(mod_name)
        except ImportError:
            missing_modules.append(mod_name)
            if i < len(packages) and packages[i]:
                missing_packages.append(packages[i])

    if missing_modules:
        if len(missing_modules) == 1:
            error_msg = f'`{missing_modules[0]}` not found.'
        else:
            error_msg = f'The following modules are not found: {", ".join(f"`{mod}`" for mod in missing_modules)}.'

        if missing_packages:
            if len(missing_packages) == 1:
                error_msg += f' Please run `pip install {missing_packages[0]}` to use {feature_name}.'
            else:
                unique_packages = list(dict.fromkeys(missing_packages))  # Remove duplicates while preserving order
                error_msg += f' Please run `pip install {" ".join(unique_packages)}` to use {feature_name}.'

        if raise_warning:
            logger.warning(error_msg)

        if not IS_BUILD_DOC and raise_error:
            raise ImportError(error_msg)
        return False

    return True


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
