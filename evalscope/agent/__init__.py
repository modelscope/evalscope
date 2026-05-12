"""Concrete implementations of the Agent evaluation framework.

Package layout mirrors ``evalscope/api/agent``:

* ``strategies/``    - registered :class:`AgentStrategy` subclasses
* ``environments/``  - registered :class:`AgentEnvironment` subclasses (T2+)
* ``tools/``         - registered tool handlers (T2+)

Registries themselves live in :mod:`evalscope.api.registry`.  Importing
this package only triggers the ``@register_*`` decorators in the
submodules below.
"""

# Trigger tool handler registration (T2).
# Trigger environment registration (T2).
# Trigger strategy registration.
from . import environments  # noqa: F401
from . import strategies  # noqa: F401
from . import tools  # noqa: F401

__all__ = []
