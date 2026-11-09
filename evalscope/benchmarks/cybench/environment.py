"""Sample-owned Docker runtime for the official Cybench task scripts."""

import asyncio
import json
import shlex
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
from evalscope.api.agent.types import ExecResult
from evalscope.api.sandbox import merge_sandbox_config_dicts
from evalscope.benchmarks.cybench.dataset import CybenchTask, stage_task_archive

WORK_DIR = '/tmp/cyber-bench'
DEFAULT_IMAGE = 'cybench/cybench@sha256:8a26be614f9d362ef7cdc61c26037884bee24af6c7ec62abf000462bb0623726'
TREE_IGNORE = '*.jpg|*.jpeg|*.png|*.gif|*.bmp|*.tiff|*.svg|*.webp|*.mp4|*.avi|*.mov|*.wmv|*.flv|*.mkv|*.webm|*.ico'


class CybenchInitialDirectory(BaseModel):
    """Upstream file-presence decision and the media-filtered directory tree."""

    model_config = ConfigDict(extra='forbid')
    files_present: bool
    tree: str


# Enable nested service limits inside the sample's private cgroup namespace.
# See https://github.com/moby/moby/blob/master/hack/dind.
CONTAINER_START = r"""
set -e
if test -f /sys/fs/cgroup/cgroup.controllers && test "$(cut -d: -f3 /proc/self/cgroup)" = /; then
    mkdir -p /sys/fs/cgroup/init
    ready=0
    for attempt in $(seq 1 30); do
        while read -r pid; do echo "$pid" > /sys/fs/cgroup/init/cgroup.procs 2>/dev/null || true; done \
            < /sys/fs/cgroup/cgroup.procs
        if sed 's/ / +/g; s/^/+/' /sys/fs/cgroup/cgroup.controllers > /sys/fs/cgroup/cgroup.subtree_control; then
            ready=1
            break
        fi
    done
    test "$ready" = 1
fi
exec sleep infinity
""".strip()


class CybenchEnvironment(EnclaveAgentEnvironment):
    """Reuse ms-enclave lifecycle and image management; keep official targets sample-local."""

    name = 'cybench'

    def __init__(
        self,
        task: CybenchTask,
        archive_path: str,
        setup_timeout: float = 1800,
        sandbox_config: Optional[Dict[str, Any]] = None,
        **env_kwargs: Any,
    ) -> None:
        self.task_path = task.path
        self.target_host = task.target_host
        self.metadata_sha256 = task.metadata_sha256
        self.archive_sha256 = task.archive_sha256
        self.archive_path = archive_path
        self.setup_timeout = setup_timeout
        self.runtime: dict = {}
        self.work_dir = WORK_DIR
        self.shell_container: Optional[str] = None
        config = merge_sandbox_config_dicts(
            {
                'image': DEFAULT_IMAGE,
                'entrypoint': ['bash', '-c', CONTAINER_START],
                'working_dir': '/',
                'privileged': True,
                'tools_config': ['shell_executor'],
            },
            sandbox_config,
        )
        self.image = config['image']
        super().__init__(**env_kwargs, sandbox_config=config)

    async def _root_exec(self, args: List[str], timeout: float = 60) -> ExecResult:
        return await super().exec(args, timeout=timeout)

    async def _checked(self, args: List[str], timeout: float = 60) -> str:
        result = await self._root_exec(args, timeout)
        if result.returncode:
            raise RuntimeError(f'Cybench initialization failed: {result.stderr}\n{result.stdout}')
        return result.stdout

    async def _runtime_identity(self) -> dict:
        handle = await self._ensure_sandbox()
        info = await handle.get_info()
        if info is None:
            raise RuntimeError('Cybench sandbox information is unavailable')

        container = await self._inspect_runtime(info.metadata['container_id'])
        return {
            'image': container['Image'],
            'cgroupns': container['HostConfig'].get('CgroupnsMode'),
            'container_id': container['Id'],
            'requested_image': self.image,
            'manager': 'ms-enclave',
            'sandbox_id': info.id,
            'memory_limit': info.config.get('memory_limit'),
            'cpu_limit': info.config.get('cpu_limit'),
        }

    async def _inspect_runtime(self, container_id: str) -> dict:
        def inspect() -> dict:
            # Read only: ms-enclave owns all host image/container lifecycle operations.
            from docker import DockerClient

            client = DockerClient.from_env()
            try:
                return client.containers.get(container_id).attrs
            finally:
                client.close()

        return await asyncio.to_thread(inspect)

    async def _check_runtime(self) -> None:
        container_id = self.runtime.get('container_id')
        if not container_id:
            raise RuntimeError('Cybench runtime is not initialized')
        container = await self._inspect_runtime(container_id)
        state = container['State']
        if not state['Running'] or state.get('Paused') or state.get('Restarting'):
            raise RuntimeError(f'Cybench runtime unavailable: {state}')
        if self.shell_container:
            terminal = await self._root_exec(['docker', 'inspect', '--format', '{{json .State}}', self.shell_container])
            if terminal.returncode or terminal.timed_out:
                raise RuntimeError(f'Cybench terminal unavailable: {terminal.stderr or terminal.stdout}')
            state = json.loads(terminal.stdout)
            if not state['Running'] or state.get('Paused') or state.get('Restarting'):
                raise RuntimeError(f'Cybench terminal unavailable: {state}')

    async def _wait_for_services(self) -> None:
        for address in self.target_host.split():
            hostname, port = address.rsplit(':', 1)
            health = await self.exec(
                [
                    'python3',
                    '-c',
                    """import socket, sys, time
deadline = time.monotonic() + 60
while True:
    try:
        socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=5).close()
        break
    except OSError:
        if time.monotonic() >= deadline:
            raise
        time.sleep(1)
""",
                    hostname,
                    port,
                ],
                timeout=70,
            )
            if health.returncode:
                raise RuntimeError(f'Cybench target service {address} is not ready: {health.stderr}')

    async def _initial_directory(self) -> CybenchInitialDirectory:
        listing = await self.exec(['ls'], timeout=60)
        if listing.returncode:
            raise RuntimeError(f'Cybench initial workspace is unavailable: {listing.stderr}')
        tree = await self.exec(['tree', '-I', TREE_IGNORE], timeout=60)
        if tree.returncode:
            raise RuntimeError(f'Cybench initial directory tree is unavailable: {tree.stderr}')
        return CybenchInitialDirectory(files_present=bool(listing.stdout), tree=tree.stdout)

    async def setup(self) -> CybenchInitialDirectory:
        """Run upstream requirements/start/init scripts before exposing the working directory."""
        try:
            self.runtime = await self._runtime_identity()
            task = f'/app/{self.task_path}'
            staging = await asyncio.to_thread(stage_task_archive, self.archive_path, self.archive_sha256)
            try:
                # Do not let files embedded in the runtime image mask an incomplete dataset package.
                await self._checked(['rm', '-rf', '--', task])
                await super().put_dir(staging.name, task)
            finally:
                staging.cleanup()
            self.runtime['archive_sha256'] = self.archive_sha256
            preparation = f"""test "$(sha256sum metadata/metadata.json | cut -d ' ' -f1)" = {shlex.quote(self.metadata_sha256)}
if test -f metadata/requirements.sh; then (cd metadata && bash requirements.sh); fi
if test -f start_docker.sh; then bash start_docker.sh; fi
bash init_script.sh {WORK_DIR}
"""
            needs_docker = 'test -f "$task/start_docker.sh" || grep -q docker "$task/init_script.sh"'
            script = f"""set -e
task={shlex.quote(task)}
mkdir -p {WORK_DIR}
if {needs_docker}; then
    if ! docker compose version >/dev/null 2>&1 && command -v docker-compose >/dev/null 2>&1; then
        mkdir -p /usr/local/lib/docker/cli-plugins
        ln -s "$(command -v docker-compose)" /usr/local/lib/docker/cli-plugins/docker-compose
    fi
    dockerd --bip=10.240.0.1/24 --default-address-pool=base=10.241.0.0/16,size=24 >/tmp/cybench-dockerd.log 2>&1 &
    for attempt in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
    docker info >/dev/null
    docker network create shared_net
fi
cd "$task"
{preparation}
"""
            setup = await self._root_exec(
                ['bash', '-c', script],
                self.setup_timeout,
            )
            self.runtime['setup'] = setup.model_dump(mode='json')
            if setup.returncode:
                raise RuntimeError(f'Cybench initialization failed: {setup.stderr}\n{setup.stdout}')
            # Resolve task aliases from the private daemon into the agent container's hosts file.
            nested = await self._root_exec(['docker', 'ps', '-aq'])
            if nested.returncode == 0:
                containers = nested.stdout.split()
                details = (
                    json.loads(
                        await self._checked(
                            [
                                'docker',
                                'inspect',
                                *containers,
                            ]
                        )
                    )
                    if containers
                    else []
                )
                for value in details:
                    for endpoint in value['NetworkSettings']['Networks'].values():
                        ip = endpoint['IPAddress']
                        if not ip:
                            continue
                        names = [value['Name'].lstrip('/'), *(endpoint.get('Aliases') or [])]
                        await self._checked(
                            [
                                'bash',
                                '-c',
                                f'printf "\\n%s %s\\n" {shlex.quote(ip)} {shlex.quote(" ".join(names))} >> /etc/hosts',
                            ]
                        )
                self.runtime['services'] = [
                    {
                        'name': value['Name'].lstrip('/'),
                        'image': value['Config']['Image'],
                        'image_id': value['Image'],
                        'state': value['State']['Status'],
                    }
                    for value in details
                ]
            # Keep service bind mounts alive, but make annotations and all source material inaccessible to the agent.
            await self._checked(
                [
                    'bash',
                    '-c',
                    f"""
set -e
id cybench-agent >/dev/null 2>&1 || useradd -m cybench-agent
chmod 700 /app /benchmark /root 2>/dev/null || true
chown -R cybench-agent:cybench-agent {WORK_DIR}
""",
                ]
            )
            initial_directory = await self._initial_directory()
            if self.task_path.endswith('/crypto/randsubware'):
                # This official task starts an appuser shell, rather than exposing a target port.
                self.shell_container = 'my-container'
                self.work_dir = '/app'
                await self._check_runtime()
            self.runtime['execution_user'] = 'appuser' if self.shell_container else 'cybench-agent'
            await self._wait_for_services()
            return initial_directory
        except BaseException:
            await self.close()
            raise

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        """Execute a command in the task's public shell with a process timeout."""
        await self._check_runtime()
        seconds = timeout or 120
        if self.shell_container:
            args = ['docker', 'exec', '-i', '-w', cwd or self.work_dir]
            for key, value in (env or {}).items():
                args += ['-e', f'{key}={value}']
            args += [self.shell_container, 'timeout', '--kill-after=5', str(seconds)]
            work_dir = '/'
        else:
            args = ['runuser', '-u', 'cybench-agent', '--']
            work_dir = cwd or self.work_dir
        result = await super().exec(
            args + cmd,
            cwd=work_dir,
            input=input,
            timeout=seconds,
            env=env,
        )
        await self._check_runtime()
        return result
