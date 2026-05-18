"""Registered AgentEnvironment implementations.

Importing this package triggers ``@register_environment`` decorators in the
submodules below.  The registries themselves live in
:mod:`evalscope.api.registry`.
"""

from .enclave import EnclaveAgentEnvironment
from .local import LocalAgentEnvironment

__all__ = ['EnclaveAgentEnvironment', 'LocalAgentEnvironment']
