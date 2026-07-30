"""Local Docker service runtime managed through ms-enclave."""

from __future__ import annotations

import asyncio
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from evalscope.api.environment import EnvironmentRuntime, EnvironmentRuntimeHandle
from evalscope.api.registry import register_environment_runtime
from evalscope.api.sandbox import SandboxEngine, SandboxHandle, get_sandbox_service
from evalscope.utils.import_utils import check_import


class MsEnclaveDockerHandle(EnvironmentRuntimeHandle):
    """Owned ms-enclave Docker sandbox exposing an HTTP endpoint."""

    name = 'ms_enclave_docker'

    def __init__(self, *, base_url: str, handle: SandboxHandle, container_id: Optional[str]) -> None:
        self.base_url = base_url
        self._handle = handle
        self._container_id = container_id
        self._closed = False

    async def capture_logs(self, destination: str | Path) -> bool:
        if not self._container_id:
            return False
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _capture() -> None:
            result = subprocess.run(
                ['docker', 'logs', self._container_id],
                capture_output=True,
                text=True,
                check=False,
            )
            content = result.stdout
            if result.stderr:
                content += f'\n[stderr]\n{result.stderr}'
            path.write_text(content, encoding='utf-8')

        await asyncio.to_thread(_capture)
        return True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._handle.close()


@register_environment_runtime('ms_enclave_docker')
class MsEnclaveDockerRuntime(EnvironmentRuntime):
    """Start one local Docker service through ms-enclave.

    This runtime deliberately fixes the ms-enclave engine to Docker. It does
    not accept Volcengine configuration.
    """

    _ALLOWED_KEYS = {'host', 'container_port', 'ready_timeout_s', 'manager_config'}

    async def start(
        self,
        *,
        image: Optional[str],
        env_vars: Dict[str, str],
        config: Dict[str, Any],
    ) -> EnvironmentRuntimeHandle:
        if not image:
            raise ValueError('ms_enclave_docker environment runtime requires an image.')
        unknown = set(config) - self._ALLOWED_KEYS
        if unknown:
            raise ValueError(f'Unsupported ms_enclave_docker options: {sorted(unknown)}')

        check_import('ms_enclave', 'evalscope[miniwob]', raise_error=True)
        from ms_enclave.sandbox.model import DockerSandboxConfig

        host = str(config.get('host', '127.0.0.1'))
        container_port = int(config.get('container_port', 8000))
        ready_timeout_s = float(config.get('ready_timeout_s', 60.0))
        if container_port <= 0 or ready_timeout_s <= 0:
            raise ValueError('container_port and ready_timeout_s must be greater than zero.')
        host_port = self._reserve_port(host)
        sandbox_config = DockerSandboxConfig(
            image=image,
            env_vars=dict(env_vars),
            ports={f'{container_port}/tcp': (host, host_port)},
            tools_config={},
            network='bridge',
            network_enabled=True,
            remove_on_exit=True,
        )
        handle = await get_sandbox_service().create_sandbox(
            SandboxEngine.DOCKER,
            sandbox_config,
            manager_config=dict(config.get('manager_config') or {}),
        )
        base_url = f'http://{host}:{host_port}'
        try:
            info = await handle.get_info()
            metadata = getattr(info, 'metadata', {}) if info is not None else {}
            container_id = metadata.get('container_id') if isinstance(metadata, dict) else None
            await asyncio.to_thread(self._wait_for_ready, base_url, ready_timeout_s)
        except Exception:
            await handle.close()
            raise
        return MsEnclaveDockerHandle(base_url=base_url, handle=handle, container_id=container_id)

    @staticmethod
    def _reserve_port(host: str) -> int:
        # Port mapping below uses an AF_INET socket; reject IPv6 hosts upfront
        # so the error is actionable instead of an opaque bind failure.
        if ':' in host:
            raise ValueError(f'ms_enclave_docker host must be an IPv4 address or hostname, got {host!r}.')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _wait_for_ready(base_url: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f'{base_url}/health', timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception as exc:
                last_error = exc
            time.sleep(0.2)
        raise TimeoutError(f'Task environment at {base_url} did not become ready within {timeout_s}s: {last_error}')


__all__ = ['MsEnclaveDockerHandle', 'MsEnclaveDockerRuntime']
