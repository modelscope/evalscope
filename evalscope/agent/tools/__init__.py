"""Registered agent tool handlers.

Importing this package triggers ``@register_agent_tool`` decorators in the
submodules below.  The registries themselves live in
:mod:`evalscope.api.registry`.
"""

from .bash import BASH_TOOL_INFO, run_bash
from .python_exec import PYTHON_EXEC_TOOL_INFO, run_python_exec
from .submit import SUBMIT_TOOL_INFO, run_submit

__all__ = [
    'BASH_TOOL_INFO',
    'PYTHON_EXEC_TOOL_INFO',
    'SUBMIT_TOOL_INFO',
    'run_bash',
    'run_python_exec',
    'run_submit',
]
