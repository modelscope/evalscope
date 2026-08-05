# Copyright (c) Alibaba, Inc. and its affiliates.
"""Instantiation and state capture for the ACEBench agent scenarios.

The simulated API classes themselves are vendored under
:mod:`evalscope.third_party.acebench`; this module owns the EvalScope-side glue that upstream keeps
in ``multi_step_utils``: picking the classes a sample involves, applying its initial state, and
capturing the attributes the checker grades.
"""
import importlib
import json
from copy import deepcopy
from typing import Any, Dict, List

VENDORED_PACKAGE = 'evalscope.third_party.acebench'

# Class name -> module holding it, relative to the vendored package.
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
        module = importlib.import_module(f'{VENDORED_PACKAGE}.{language}.{module_name}')
        instance = getattr(module, class_name)()
        # Upstream applies the class-specific config and then the BaseApi config on top.
        instance._load_scenario(deepcopy(initial_config.get(class_name, {})), long_context=False)
        instance._load_scenario(deepcopy(initial_config.get('BaseApi', {})), long_context=False)
        instances[class_name] = instance
    return instances


def snapshot_states(instances: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Capture the graded attributes of each instance as ``[{ClassName: {attr: value}}]``.

    The snapshot is round-tripped through JSON because the official pipeline writes the recorded
    state to a result file and reads it back before grading. That conversion matters: the message
    inbox and reminder list are keyed by integers in memory but by strings in the expected answer.
    """
    states = []
    for class_name, instance in instances.items():
        graded = SAVED_ATTRIBUTES.get(class_name, [])
        states.append({class_name: {name: value for name, value in vars(instance).items() if name in graded}})
    return json.loads(json.dumps(states, default=str))
