# Copyright (c) Alibaba, Inc. and its affiliates.
"""Service-wide plugin configuration shared by request validation and spawned workers."""

from typing import Any, List

from flask import current_app, has_app_context

from evalscope.plugins import normalize_plugin_names

#: ``app.config`` key holding the plugin names the operator enabled at startup.
ENABLED_PLUGINS_KEY = 'EVALSCOPE_PLUGINS'


def enabled_plugins() -> List[str]:
    """Return the plugin names enabled for this service, in load order; none outside an app."""
    if not has_app_context():
        return []
    return list(current_app.config.get(ENABLED_PLUGINS_KEY) or [])


def resolve_request_plugins(requested: Any) -> List[str]:
    """Return the plugins a task worker must load for a request.

    The service loads every enabled plugin at startup and validates requests against the
    resulting registries. Workers load the same list, so a name that passed validation
    always resolves in the worker. A request may list plugins only to require that they
    are enabled; it cannot enable new ones.

    Raises:
        ValueError: if the request names a plugin the operator did not enable.
    """
    enabled = enabled_plugins()
    names = normalize_plugin_names(requested)
    missing = [name for name in names if name not in enabled]
    if missing:
        raise ValueError(
            f'Plugins not enabled on this service: {", ".join(missing)}. '
            f'Enabled plugins: {", ".join(enabled) or "none"}.'
        )
    return enabled
