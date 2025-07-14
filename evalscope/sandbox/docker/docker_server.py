# sandbox_server.py
import asyncio
import base64
import docker
import io
import os
import tarfile
import tempfile
import uuid
from docker.errors import ImageNotFound, NotFound
from fastapi import BackgroundTasks, HTTPException
from typing import Any, Dict, List, Optional, Union

from evalscope.sandbox.config import DockerContainerConfig
from evalscope.sandbox.sandbox_server import (CodeOutput, ContainerInfo, ExecuteCodeRequest, ExecuteCommandRequest,
                                              FileOperationRequest, SandboxServer, WriteFileRequest)
from evalscope.utils import get_logger

# Configure logging
logger = get_logger(name='docker-sandbox')


class DockerSandboxServer(SandboxServer):
    """Docker-based sandbox server implementation."""

    def __init__(self):
        self.client = None
        self.containers: Dict[str, docker.models.containers.Container] = {}
        super().__init__()

    async def startup(self):
        """Initialize Docker client and resources."""
        logger.info(os.environ.get('DOCKER_HOST', 'Not set'))
        self.client = docker.from_env()

    async def shutdown(self):
        """Cleanup all running containers."""
        for container_id, container in list(self.containers.items()):
            try:
                container.stop(timeout=1)
                container.remove()
                logger.info(f'Container {container_id} cleaned up')
            except Exception as e:
                logger.error(f'Error cleaning up container {container_id}: {e}')

    async def create_container_workspace(self, container, working_dir: str):
        """Create workspace directory in container if it doesn't exist."""
        try:
            exit_code, _ = await asyncio.to_thread(container.exec_run, f'mkdir -p {working_dir}', user='root')
            if exit_code != 0:
                logger.warning(f'Failed to create workspace directory: {working_dir}')
        except Exception as e:
            logger.error(f'Error creating workspace directory: {e}')

    async def create_container(self, config: DockerContainerConfig, background_tasks: BackgroundTasks) -> ContainerInfo:
        """Create a new Docker container for sandbox execution."""
        try:
            # Check if image exists locally or pull it
            try:
                self.client.images.get(config.image)
            except ImageNotFound:
                logger.info(f'Image {config.image} not found locally. Pulling...')
                self.client.images.pull(config.image)

            # Generate unique container ID
            container_id = f'sandbox-{uuid.uuid4().hex[:8]}'

            # Prepare container creation parameters
            container_params = {
                'image': config.image,
                'command': 'tail -f /dev/null',  # Keep container running
                'hostname': 'sandbox',
                'working_dir': config.working_dir,
                'environment': config.env_vars,
                'name': container_id,
                'tty': True,
                'detach': True,
                'mem_limit': config.memory_limit,
                'cpu_period': 100000,
                'cpu_quota': int(100000 * config.cpu_limit),
                'network_mode': 'none' if not config.network_enabled else 'bridge',
            }

            # Create container
            container = self.client.containers.create(**container_params)

            # Start container
            container.start()

            # Store container reference
            self.containers[container_id] = container

            # Create workspace directory in background
            background_tasks.add_task(self.create_container_workspace, container, config.working_dir)

            # Get container information
            ports = {}
            container_info = container.attrs
            if 'NetworkSettings' in container_info and 'Ports' in container_info['NetworkSettings']:
                for container_port, host_ports in container_info['NetworkSettings']['Ports'].items():
                    if host_ports:
                        ports[container_port] = f"{host_ports[0]['HostIp']}:{host_ports[0]['HostPort']}"

            return ContainerInfo(container_id=container_id, status=container.status, image=config.image, ports=ports)

        except Exception as e:
            logger.error(f'Error creating container: {e}')
            raise HTTPException(status_code=500, detail=f'Failed to create container: {str(e)}')

    async def delete_container(self, container_id: str):
        """Delete a Docker container."""
        try:
            if container_id in self.containers:
                container = self.containers[container_id]
                container.stop(timeout=5)
                container.remove()
                del self.containers[container_id]
                logger.info(f'Container {container_id} deleted')
                return {'message': f'Container {container_id} deleted'}
            else:
                raise HTTPException(status_code=404, detail=f'Container {container_id} not found')
        except Exception as e:
            logger.error(f'Error deleting container: {e}')
            raise HTTPException(status_code=500, detail=f'Failed to delete container: {str(e)}')

    async def execute_code(self, request: ExecuteCodeRequest) -> CodeOutput:
        """Execute Python code in a Docker container."""
        temp_file = None
        try:
            if request.container_id not in self.containers:
                raise HTTPException(status_code=404, detail=f'Container {request.container_id} not found')

            container = self.containers[request.container_id]

            # Create temporary file on host
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
            temp_file.write(request.code)
            temp_file.close()

            # Generate unique target path in container
            unique_id = uuid.uuid4().hex
            target_filename = f'code_to_execute_{unique_id}.py'
            target_path = f'/tmp/{target_filename}'

            try:
                # Copy file to container using tar
                with open(temp_file.name, 'rb') as f:
                    tar_stream = io.BytesIO()
                    with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                        tarinfo = tarfile.TarInfo(name=target_filename)
                        tarinfo.size = os.path.getsize(temp_file.name)
                        tarinfo.mode = 0o644
                        tar.addfile(tarinfo, f)

                    tar_stream.seek(0)
                    await asyncio.to_thread(container.put_archive, '/tmp', tar_stream.getvalue())

                # Execute the code
                working_dir = request.working_dir or '/sandbox'
                env_vars = request.env or {}
                timeout = getattr(request, 'timeout', 30)

                logger.debug(f'Executing code in container {request.container_id}')
                logger.debug(f'Working dir: {working_dir}')
                logger.debug(f'Code length: {len(request.code)} characters')
                logger.debug(f'Timeout: {timeout} seconds')

                try:
                    exec_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            container.exec_run,
                            cmd=['python3', target_path],
                            environment=env_vars,
                            workdir=working_dir,
                            demux=True,
                            stream=False,
                        ),
                        timeout=timeout)
                except asyncio.TimeoutError:
                    logger.error(f'Code execution timed out after {timeout} seconds')
                    return CodeOutput(
                        output=None, logs='', error=f'Execution timed out after {timeout} seconds', status_code=124)

                # Handle the output properly
                if isinstance(exec_result.output, tuple):
                    stdout = exec_result.output[0] or b''
                    stderr = exec_result.output[1] or b''
                else:
                    stdout = exec_result.output or b''
                    stderr = b''

                stdout_str = stdout.decode('utf-8', errors='replace') if stdout else ''
                stderr_str = stderr.decode('utf-8', errors='replace') if stderr else ''

                logger.debug(f'Execution completed with exit code: {exec_result.exit_code}')

                return CodeOutput(
                    output=stdout_str,
                    logs=stderr_str,
                    status_code=exec_result.exit_code or 0,
                    error=stderr_str if exec_result.exit_code != 0 else None)

            finally:
                # Clean up container temp file
                try:
                    await asyncio.to_thread(container.exec_run, f'rm -f {target_path}')
                except Exception:
                    pass  # Ignore cleanup errors

        except Exception as e:
            logger.error(f'Error executing code: {e}', exc_info=True)
            return CodeOutput(output=None, logs='', error=str(e), status_code=1)
        finally:
            # Clean up host temp file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass  # Ignore cleanup errors

    async def execute_command(self, request: ExecuteCommandRequest) -> CodeOutput:
        """Execute a shell command in a Docker container."""
        try:
            if request.container_id not in self.containers:
                raise HTTPException(status_code=404, detail=f'Container {request.container_id} not found')

            container = self.containers[request.container_id]

            # Prepare environment variables
            env_vars = request.env or {}
            timeout = getattr(request, 'timeout', 30)  # Default 30 seconds

            # Execute the command
            working_dir = request.working_dir or '/sandbox'

            logger.debug(f'Executing command in container {request.container_id}')
            logger.debug(f'Command: {request.command}')
            logger.debug(f'Timeout: {timeout} seconds')

            try:
                exec_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        container.exec_run,
                        cmd=request.command,
                        environment=env_vars,
                        workdir=working_dir,
                        demux=True,
                        stream=False,
                    ),
                    timeout=timeout)
            except asyncio.TimeoutError:
                logger.error(f'Command execution timed out after {timeout} seconds')
                return CodeOutput(
                    output=None,
                    logs='',
                    error=f'Command execution timed out after {timeout} seconds',
                    status_code=124  # Standard timeout exit code
                )

            stdout = exec_result.output[0] or b''
            stderr = exec_result.output[1] or b''

            return CodeOutput(
                output=stdout.decode('utf-8', errors='replace'),
                logs=stderr.decode('utf-8', errors='replace'),
                status_code=exec_result.exit_code or 0,
                error=stderr.decode('utf-8', errors='replace') if exec_result.exit_code != 0 else None)

        except Exception as e:
            logger.error(f'Error executing command: {e}')
            return CodeOutput(output=None, logs='', error=str(e), status_code=1)

    async def read_file(self, request: FileOperationRequest):
        """Read a file from a Docker container."""
        try:
            if request.container_id not in self.containers:
                raise HTTPException(status_code=404, detail=f'Container {request.container_id} not found')

            container = self.containers[request.container_id]

            # Get file from container
            try:
                bits, stat = await asyncio.to_thread(container.get_archive, request.path)

                # Extract file content from tar archive
                with tempfile.NamedTemporaryFile() as tmp:
                    for chunk in bits:
                        tmp.write(chunk)
                    tmp.seek(0)

                    with tarfile.open(fileobj=tmp) as tar:
                        member = tar.next()
                        if not member:
                            raise ValueError('Empty tar archive')

                        file_content = tar.extractfile(member)
                        if not file_content:
                            raise ValueError('Failed to extract file content')

                        content = file_content.read()

                        # Try to decode as text if possible
                        try:
                            return {'content': content.decode('utf-8'), 'binary': False}
                        except UnicodeDecodeError:
                            # Return base64 encoded content if it's binary
                            return {'content': base64.b64encode(content).decode('ascii'), 'binary': True}

            except NotFound:
                raise HTTPException(status_code=404, detail=f'File not found: {request.path}')

        except Exception as e:
            logger.error(f'Error reading file: {e}')
            raise HTTPException(status_code=500, detail=f'Failed to read file: {str(e)}')

    async def write_file(self, request: WriteFileRequest):
        """Write a file to a Docker container."""
        try:
            if request.container_id not in self.containers:
                raise HTTPException(status_code=404, detail=f'Container {request.container_id} not found')

            container = self.containers[request.container_id]

            # Decode content if binary
            if request.binary:
                content = base64.b64decode(request.content)
            else:
                content = request.content.encode('utf-8')

            # Create parent directory if needed
            parent_dir = os.path.dirname(request.path)
            if parent_dir:
                await asyncio.to_thread(container.exec_run, f'mkdir -p {parent_dir}')

            # Create tar archive with file
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tarinfo = tarfile.TarInfo(name=os.path.basename(request.path))
                tarinfo.size = len(content)
                tar.addfile(tarinfo, io.BytesIO(content))

            tar_stream.seek(0)

            # Copy file to container
            await asyncio.to_thread(container.put_archive, os.path.dirname(request.path) or '/', tar_stream.getvalue())

            return {'message': f'File {request.path} written successfully'}

        except Exception as e:
            logger.error(f'Error writing file: {e}')
            raise HTTPException(status_code=500, detail=f'Failed to write file: {str(e)}')


if __name__ == '__main__':
    # Create server instance
    server = DockerSandboxServer()
    server.run()
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
