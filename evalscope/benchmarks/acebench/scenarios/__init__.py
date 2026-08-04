# Copyright (c) Alibaba, Inc. and its affiliates.
"""Simulated APIs used by ACEBench agent tasks.

The modules under ``en/`` and ``zh/`` are vendored verbatim from the official ACEBench repository
(https://github.com/ACEBench/ACEBench, MIT License), ``model_inference/multi_step/scenarios{en,zh}``,
with only the ``BaseApi`` import rewritten to be relative. ACEBench is not published as a package,
and the agent categories are graded on the state these classes end up in, so they have to be
executed as-is for the scores to mean the same thing.

The classes are plain in-memory state machines: they touch no files, network or subprocesses, which
is why the rollout executes them in-process instead of inside a code sandbox. Model output never
reaches ``eval``; see :mod:`evalscope.benchmarks.acebench.rollout` for the dispatch.
"""
import importlib
from copy import deepcopy
from typing import Any, Dict, List

# Class name -> module holding it, relative to this package.
CLASS_MODULE_MAPPING = {
    'BaseApi': 'base_api',
    'MessageApi': 'message',
    'ReminderApi': 'reminder',
    'FoodPlatform': 'food_services',
    'Travel': 'travel',
}

# Attributes compared against the expected end state, see SAVED_CLASS upstream.
SAVED_ATTRIBUTES = {
    'BaseApi': ['wifi', 'logged_in'],
    'MessageApi': ['inbox'],
    'ReminderApi': ['reminder_list'],
    'FoodPlatform': ['users', 'logged_in_users', 'orders'],
    'Travel': ['users', 'reservations'],
}


def load_scenario_instances(
    initial_config: Dict[str, Any],
    involved_classes: List[str],
    language: str,
) -> Dict[str, Any]:
    """Instantiate the simulated APIs a sample involves and apply its initial state.

    Args:
        initial_config: Per-class initial state taken from the sample.
        involved_classes: Names of the API classes the sample uses.
        language: Dataset language, ``en`` or ``zh``.

    Returns:
        Mapping of class name to a freshly initialized instance.
    """
    instances = {}
    for class_name in involved_classes:
        module_name = CLASS_MODULE_MAPPING.get(class_name)
        if module_name is None:
            raise ValueError(f'Unknown ACEBench scenario class: {class_name}')
        module = importlib.import_module(f'{__name__}.{language}.{module_name}')
        instance = getattr(module, class_name)()
        # Upstream applies the class-specific config and then the BaseApi config on top.
        instance._load_scenario(deepcopy(initial_config.get(class_name, {})), long_context=False)
        instance._load_scenario(deepcopy(initial_config.get('BaseApi', {})), long_context=False)
        instances[class_name] = instance
    return instances


def snapshot_states(instances: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Capture the graded attributes of each instance as ``[{ClassName: {attr: value}}]``."""
    states = []
    for class_name, instance in instances.items():
        graded = SAVED_ATTRIBUTES.get(class_name, [])
        states.append({class_name: {name: value for name, value in vars(instance).items() if name in graded}})
    return states
