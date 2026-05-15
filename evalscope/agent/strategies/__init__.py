"""Registered agent strategies.

Importing this package triggers ``@register_strategy`` decorators.
"""

from . import swe_bench  # noqa: F401 - register SWE-bench strategies
from .function_calling import FunctionCallingStrategy
from .react import ReactStrategy
from .swe_bench import SweBenchBackticksStrategy, SweBenchToolcallStrategy

__all__ = [
    'FunctionCallingStrategy',
    'ReactStrategy',
    'SweBenchToolcallStrategy',
    'SweBenchBackticksStrategy',
]
