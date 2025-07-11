import requests
from typing import Any, Dict, List, Optional, Union

from evalscope.sandbox.config import DockerContainerConfig
from evalscope.sandbox.sandbox_client import SandboxClient


class DockerSandboxClient(SandboxClient):
    """Client for interacting with Docker Sandbox Server API."""

    def __init__(self, base_url: str = 'http://localhost:8000', config: Optional[DockerContainerConfig] = None):
        """Initialize the Docker Sandbox Client.

        Args:
            base_url: The base URL of the sandbox server
            config: Optional Docker container configuration for context manager
        """
        super().__init__(base_url)
        self.config = config

    def create_container(self, config: DockerContainerConfig) -> str:
        """Create a new Docker container.

        Args:
            config: Docker container configuration object

        Returns:
            Container ID
        """
        response = requests.post(f"{self.base_url}/container/create", json=config.model_dump())

        if response.status_code != 200:
            raise Exception(f"Failed to create container: {response.text}")

        data = response.json()
        self.container_id = data['container_id']
        return self.container_id

    def delete_container(self):
        """Delete the current Docker container."""
        if not self.container_id:
            raise ValueError('No container ID available')

        response = requests.post(f"{self.base_url}/container/delete", params={'container_id': self.container_id})

        if response.status_code != 200:
            raise Exception(f"Failed to delete container: {response.text}")

        result = response.json()
        self.container_id = None
        return result

    def execute_code(self,
                     code: str,
                     working_dir: str = None,
                     timeout: int = None,
                     env: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute Python code in the container.

        Args:
            code: Python code to execute
            working_dir: Working directory for execution
            timeout: Timeout for execution
            env: Environment variables for execution

        Returns:
            Dictionary with execution results
        """
        if not self.container_id:
            raise ValueError('No container ID available')

        request_data = {
            'container_id': self.container_id,
            'code': code,
            'working_dir': working_dir,
            'timeout': timeout,
            'env': env or {}
        }

        response = requests.post(f"{self.base_url}/execute/code", json=request_data)

        if response.status_code != 200:
            raise Exception(f"Failed to execute code: {response.text}")

        return response.json()

    def execute_command(self,
                        command: List[str],
                        working_dir: str = None,
                        timeout: int = None,
                        env: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute a shell command in the container.

        Args:
            command: Command to execute as a list of strings
            working_dir: Working directory for execution
            timeout: Timeout for execution
            env: Environment variables for execution

        Returns:
            Dictionary with execution results
        """
        if not self.container_id:
            raise ValueError('No container ID available')

        request_data = {
            'container_id': self.container_id,
            'command': command,
            'working_dir': working_dir,
            'timeout': timeout,
            'env': env or {}
        }

        response = requests.post(f"{self.base_url}/execute/command", json=request_data)

        if response.status_code != 200:
            raise Exception(f"Failed to execute command: {response.text}")

        return response.json()

    def read_file(self, path: str) -> Union[str, bytes]:
        """Read a file from the container.

        Args:
            path: Path to file in container

        Returns:
            File content (string for text files, bytes for binary files)
        """
        if not self.container_id:
            raise ValueError('No container ID available')

        request_data = {'container_id': self.container_id, 'path': path}

        response = requests.post(f"{self.base_url}/file/read", json=request_data)

        if response.status_code != 200:
            raise Exception(f"Failed to read file: {response.text}")

        data = response.json()
        if data['binary']:
            import base64
            return base64.b64decode(data['content'])
        else:
            return data['content']

    def write_file(self, path: str, content: Union[str, bytes]) -> Dict[str, Any]:
        """Write a file to the container.

        Args:
            path: Path to file in container
            content: Content to write (string or bytes)

        Returns:
            Response data
        """
        if not self.container_id:
            raise ValueError('No container ID available')

        binary = isinstance(content, bytes)
        if binary:
            import base64
            content = base64.b64encode(content).decode('ascii')

        request_data = {'container_id': self.container_id, 'path': path, 'content': content, 'binary': binary}

        response = requests.post(f"{self.base_url}/file/write", json=request_data)

        if response.status_code != 200:
            raise Exception(f"Failed to write file: {response.text}")

        return response.json()

    def __enter__(self):
        """Context manager entry - creates a container if none exists."""
        if self.container_id is None:
            # Use stored config or create default Docker configuration
            config = self.config or DockerContainerConfig(image='sandbox-image')
            self.create_container(config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - deletes the container."""
        if self.container_id:
            try:
                self.delete_container()
            except Exception:
                pass
