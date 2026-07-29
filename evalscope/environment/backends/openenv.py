"""OpenEnv protocol backend."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Dict

from evalscope.api.environment import EnvironmentStepResult, TaskEnvironmentBackend, TaskEnvironmentSession
from evalscope.api.registry import register_task_environment
from evalscope.utils.import_utils import check_import


class OpenEnvSession(TaskEnvironmentSession):
    """Lazily connected OpenEnv simulation session."""

    backend_name = 'openenv'

    def __init__(
        self,
        *,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        max_message_size_mb: float = 100.0,
    ) -> None:
        if not base_url:
            raise ValueError('OpenEnv base_url must not be empty.')
        if connect_timeout_s <= 0 or message_timeout_s <= 0 or max_message_size_mb <= 0:
            raise ValueError('OpenEnv timeout and message-size values must be greater than zero.')
        self._base_url = base_url.rstrip('/')
        self._connect_timeout_s = connect_timeout_s
        self._message_timeout_s = message_timeout_s
        self._max_message_size_mb = max_message_size_mb
        self._client: Any = None
        self._connect_lock = asyncio.Lock()
        self._closed = False

    async def reset(self, **kwargs: Any) -> EnvironmentStepResult:
        client = await self._ensure_client()
        return self._normalize_result(await client.reset(**kwargs))

    async def step(self, action: Dict[str, Any]) -> EnvironmentStepResult:
        client = await self._ensure_client()
        return self._normalize_result(await client.step(action))

    async def state(self) -> Dict[str, Any]:
        client = await self._ensure_client()
        state = await client.state()
        if not isinstance(state, dict):
            raise TypeError(f'OpenEnv state must be a dictionary, got {type(state).__name__}.')
        return state

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    async def _ensure_client(self) -> Any:
        if self._closed:
            raise RuntimeError('OpenEnv session is already closed.')
        if self._client is not None:
            return self._client
        async with self._connect_lock:
            if self._client is not None:
                return self._client
            check_import('openenv', extra='miniwob', raise_error=True, feature_name='OpenEnv task environment')
            from openenv.core import GenericEnvClient

            client = GenericEnvClient(
                base_url=self._base_url,
                connect_timeout_s=self._connect_timeout_s,
                message_timeout_s=self._message_timeout_s,
                max_message_size_mb=self._max_message_size_mb,
                mode='simulation',
            )
            try:
                await client.connect()
            except Exception:
                with contextlib.suppress(Exception):
                    await client.close()
                raise
            self._client = client
            return client

    @staticmethod
    def _normalize_result(result: Any) -> EnvironmentStepResult:
        observation = getattr(result, 'observation', {})
        if not isinstance(observation, dict):
            raise TypeError(f'OpenEnv observation must be a dictionary, got {type(observation).__name__}.')
        metadata = getattr(result, 'metadata', None)
        return EnvironmentStepResult(
            observation=observation,
            reward=getattr(result, 'reward', None),
            done=bool(getattr(result, 'done', False)),
            metadata=metadata if isinstance(metadata, dict) else {},
        )


@register_task_environment('openenv')
class OpenEnvBackend(TaskEnvironmentBackend):
    """Create OpenEnv sessions without owning service/container lifecycle."""

    _ALLOWED_KEYS = {'connect_timeout_s', 'message_timeout_s', 'max_message_size_mb'}

    def create_session(
        self,
        *,
        base_url: str,
        config: Dict[str, Any],
    ) -> TaskEnvironmentSession:
        unknown = set(config) - self._ALLOWED_KEYS
        if unknown:
            raise ValueError(f'Unsupported OpenEnv backend options: {sorted(unknown)}')
        return OpenEnvSession(base_url=base_url, **config)


__all__ = ['OpenEnvBackend', 'OpenEnvSession']
