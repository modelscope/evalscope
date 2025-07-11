from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .config import ContainerConfig


class SandboxClient(ABC):
    """Abstract base class for all sandbox clients."""

    def __init__(self, base_url: str = None):
        """Initialize the sandbox client.

        Args:
            base_url: The base URL of the sandbox server (if applicable)
        """
        self.base_url = base_url
        self.container_id = None

    @abstractmethod
    def create_container(self, config: ContainerConfig) -> str:
        """Create a new container/sandbox environment.

        Args:
            config: Container configuration object

        Returns:
            Container/environment ID
        """
        pass

    @abstractmethod
    def delete_container(self):
        """Delete the current container/sandbox environment."""
        pass

    @abstractmethod
    def execute_code(self,
                     code: str,
                     working_dir: str = None,
                     timeout: int = None,
                     env: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute code in the sandbox.

        Args:
            code: Code to execute
            working_dir: Working directory for execution
            timeout: Timeout for execution
            env: Environment variables for execution

        Returns:
            Dictionary with execution results
        """
        pass

    @abstractmethod
    def execute_command(self,
                        command: List[str],
                        working_dir: str = None,
                        timeout: int = None,
                        env: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute a shell command in the sandbox.

        Args:
            command: Command to execute as a list of strings
            working_dir: Working directory for execution
            timeout: Timeout for execution
            env: Environment variables for execution

        Returns:
            Dictionary with execution results
        """
        pass

    @abstractmethod
    def read_file(self, path: str) -> Union[str, bytes]:
        """Read a file from the sandbox.

        Args:
            path: Path to file in container

        Returns:
            File content (string for text files, bytes for binary files)
        """
        pass

    @abstractmethod
    def write_file(self, path: str, content: Union[str, bytes]) -> Dict[str, Any]:
        """Write a file to the sandbox.

        Args:
            path: Path to file in container
            content: Content to write (string or bytes)

        Returns:
            Response data
        """
        pass

    def __enter__(self):
        """Context manager entry - creates a container if none exists."""
        if self.container_id is None:
            # Create with default configuration - subclasses should override if needed
            from .config import ContainerConfig
            default_config = ContainerConfig(image='sandbox-image')
            self.create_container(default_config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - deletes the container."""
        if self.container_id:
            try:
                self.delete_container()
            except Exception:
                pass
