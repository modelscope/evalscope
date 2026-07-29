"""Runtime for an already hosted task-environment service."""

from typing import Any, Dict, Optional

from evalscope.api.environment import EnvironmentRuntime, EnvironmentRuntimeLease
from evalscope.api.registry import register_environment_runtime


class RemoteEnvironmentLease(EnvironmentRuntimeLease):
    """Non-owning lease for a trusted remote service endpoint."""

    name = 'remote'
    is_local = False

    def __init__(self, base_url: str) -> None:
        if not base_url:
            raise ValueError('Remote environment runtime requires base_url.')
        self.base_url = base_url.rstrip('/')

    async def close(self) -> None:
        return None


@register_environment_runtime('remote')
class RemoteEnvironmentRuntime(EnvironmentRuntime):
    """Resolve a trusted remote task-environment URL."""

    name = 'remote'

    async def start(
        self,
        *,
        image: Optional[str],
        env_vars: Dict[str, str],
        config: Dict[str, Any],
    ) -> EnvironmentRuntimeLease:
        if image:
            raise ValueError('Remote environment runtime does not accept an image.')
        if env_vars:
            raise ValueError('Remote environment runtime does not accept env_vars.')
        unknown = set(config) - {'base_url'}
        if unknown:
            raise ValueError(f'Unsupported remote environment runtime options: {sorted(unknown)}')
        return RemoteEnvironmentLease(base_url=str(config.get('base_url', '')))


__all__ = ['RemoteEnvironmentLease', 'RemoteEnvironmentRuntime']
