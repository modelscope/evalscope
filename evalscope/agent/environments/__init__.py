"""Registered AgentEnvironment implementations.

Importing this package triggers ``@register_environment`` decorators in the
submodules below. The registry itself lives in
:mod:`evalscope.api.registry`.
"""

from .enclave import EnclaveAgentEnvironment
from .local import LocalAgentEnvironment, TemporaryLocalAgentEnvironment

__all__ = ['EnclaveAgentEnvironment', 'LocalAgentEnvironment', 'TemporaryLocalAgentEnvironment']
