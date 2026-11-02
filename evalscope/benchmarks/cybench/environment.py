"""Sample-owned Docker runtime for the official Cybench task scripts."""

import asyncio
import hashlib
import json
import shlex
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.agent.types import ExecResult
from evalscope.benchmarks.cybench.sources import GLACIER_REPO, GlacierTask, stage_glacier_task

WORK_DIR = '/tmp/cyber-bench'
DEFAULT_IMAGE = 'cybench/cybench@sha256:8a26be614f9d362ef7cdc61c26037884bee24af6c7ec62abf000462bb0623726'


class CybenchEnvironment(AgentEnvironment):
    """Use a private Docker daemon per sample; never mount the host Docker socket or credentials."""

    name = 'cybench'

    def __init__(
        self,
        task_path: str,
        metadata_sha256: str,
        image: str = DEFAULT_IMAGE,
        source_root: Optional[str] = None,
        setup_timeout: float = 1800,
        glacier: Optional[GlacierTask] = None,
        glacier_source_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.task_path = task_path
        self.metadata_sha256 = metadata_sha256
        self.image = image
        self.source_root = source_root
        self.setup_timeout = setup_timeout
        self.container = f'evalscope-cybench-{uuid.uuid4().hex}'
        self.started = False
        self.runtime: dict = {}
        self.work_dir = WORK_DIR
        self.shell_container: Optional[str] = None
        self.glacier = glacier
        self.glacier_source_root = glacier_source_root
        self.cache_dir = cache_dir or str(Path.home() / '.cache/evalscope')

    async def _docker(self, args: List[str], timeout: float = 60, input: Optional[str] = None) -> ExecResult:
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            'docker',
            *args,
            stdin=asyncio.subprocess.PIPE if input is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input.encode() if input is not None else None),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            raise TimeoutError(f'Docker operation timed out after {timeout}s: {args[:2]}')
        return ExecResult(
            stdout=stdout.decode('utf-8', errors='replace'),
            stderr=stderr.decode('utf-8', errors='replace'),
            returncode=process.returncode,
            duration=time.monotonic() - started,
        )

    async def _checked(self, args: List[str], timeout: float = 60) -> str:
        result = await self._docker(args, timeout)
        if result.returncode:
            raise RuntimeError(f'Cybench Docker operation failed: {result.stderr}\n{result.stdout}')
        return result.stdout

    async def setup(self) -> str:
        """Run upstream requirements/start/init scripts before exposing the working directory."""
        try:
            self.started = True
            await self._checked(
                [
                    'run',
                    '-d',
                    '--name',
                    self.container,
                    '--privileged',
                    '--entrypoint',
                    'sleep',
                    self.image,
                    'infinity',
                ],
                self.setup_timeout,
            )
            self.runtime = json.loads(await self._checked(['inspect', self.container]))[0]
            self.runtime = {'image': self.runtime['Image'], 'requested_image': self.image}
            task = f'/app/{self.task_path}'
            if self.glacier:
                staging = await asyncio.to_thread(
                    stage_glacier_task,
                    self.glacier,
                    self.cache_dir,
                    self.glacier_source_root,
                )
                try:
                    await self._checked(['exec', self.container, 'mkdir', '-p', task])
                    await self._checked(['cp', f'{staging.name}/.', f'{self.container}:{task}'], self.setup_timeout)
                finally:
                    staging.cleanup()
                self.runtime['task_source'] = {
                    'repository': GLACIER_REPO,
                    'revision': self.glacier.revision,
                    'reference': self.glacier.reference.model_dump(mode='json'),
                    'files': [file.model_dump(mode='json') for file in self.glacier.files],
                }
            elif self.source_root:
                source = (Path(self.source_root) / self.task_path).resolve()
                if not source.is_relative_to(Path(self.source_root).resolve()):
                    raise ValueError('Cybench task path escapes the source checkout.')
                if hashlib.sha256((source / 'metadata/metadata.json').read_bytes()).hexdigest() != self.metadata_sha256:
                    raise ValueError('Cybench task metadata differs from the pinned source.')
                await self._checked(['exec', self.container, 'mkdir', '-p', task])
                await self._checked(['cp', f'{source}/.', f'{self.container}:{task}'], self.setup_timeout)
            if self.glacier:
                preparation = ''
                if self.glacier.service:
                    preparation += 'docker compose up --build -d\n'
                for file in self.glacier.files:
                    if file.target:
                        destination = f'{WORK_DIR}/{file.target}'
                        preparation += f'mkdir -p {shlex.quote(str(Path(destination).parent))}\n'
                        preparation += f'cp -- {shlex.quote(file.path)} {shlex.quote(destination)}\n'
                needs_docker = 'true' if self.glacier.service else 'false'
            else:
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
    dockerd --bip=10.240.0.1/24 --default-address-pool=base=10.241.0.0/16,size=24 >/tmp/cybench-dockerd.log 2>&1 &
    for attempt in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
    docker info >/dev/null
    docker network create shared_net
fi
cd "$task"
{preparation}
"""
            setup = await self._docker(
                ['exec', self.container, 'bash', '-c', script],
                self.setup_timeout,
            )
            self.runtime['setup'] = setup.model_dump(mode='json')
            if setup.returncode:
                raise RuntimeError(f'Cybench initialization failed: {setup.stderr}\n{setup.stdout}')
            # Resolve task aliases from the private daemon into the agent container's hosts file.
            nested = await self._docker(['exec', self.container, 'docker', 'ps', '-aq'])
            if nested.returncode == 0:
                containers = nested.stdout.split()
                details = (
                    json.loads(
                        await self._checked(
                            [
                                'exec',
                                self.container,
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
                                'exec',
                                self.container,
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
                    'exec',
                    self.container,
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
            if self.task_path.endswith('/crypto/randsubware'):
                # This official task starts an appuser shell, rather than exposing a target port.
                self.shell_container = 'my-container'
                self.work_dir = '/app'
            self.runtime['execution_user'] = 'appuser' if self.shell_container else 'cybench-agent'
            listing = await self.exec(['ls', '-R'])
            if listing.returncode:
                raise RuntimeError(f'Cybench initial workspace is unavailable: {listing.stderr}')
            return listing.stdout
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
        seconds = timeout or 120
        if self.shell_container:
            args = ['exec', '-i', self.container, 'docker', 'exec', '-i', '-w', cwd or self.work_dir]
            target = self.shell_container
        else:
            args = ['exec', '-i', '-u', 'cybench-agent', '-w', cwd or self.work_dir]
            target = self.container
        for key, value in (env or {}).items():
            args += ['-e', f'{key}={value}']
        result = await self._docker(
            args + [target, 'timeout', '--kill-after=5', str(seconds), *cmd],
            seconds + 15,
            input,
        )
        result.timed_out = result.returncode == 124
        return result

    async def close(self) -> None:
        """Remove only this sample's container, including its private target containers and networks."""
        if self.started:
            result = await self._docker(['rm', '-f', '-v', self.container])
            if result.returncode and 'No such container' not in result.stderr:
                raise RuntimeError(f'Cybench cleanup failed: {result.stderr}')
            self.started = False
