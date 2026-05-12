"""Registered agent strategies.

Importing this package triggers ``@register_strategy`` decorators.
"""

from .function_calling import FunctionCallingStrategy
from .mini_swe import MiniSweStrategy
from .react import ReactStrategy

__all__ = ['FunctionCallingStrategy', 'MiniSweStrategy', 'ReactStrategy']
