"""Registered agent strategies.

Importing this package triggers ``@register_strategy`` decorators.
"""

from .function_calling import FunctionCallingStrategy

__all__ = ['FunctionCallingStrategy']
