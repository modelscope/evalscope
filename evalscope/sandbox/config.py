from pydantic import BaseModel, Field
from typing import Dict, Optional


class ContainerConfig(BaseModel):
    """Base container configuration class."""
    timeout: int = 30


class DockerContainerConfig(ContainerConfig):
    """Docker-specific container configuration."""
    image: str = 'sandbox-image:latest'
    working_dir: str = '/sandbox'
    env_vars: Dict[str, str] = Field(default_factory=dict)
    volumes: Dict[str, str] = Field(default_factory=dict)
    memory_limit: str = '1g'
    cpu_limit: float = 2.0
    network_enabled: bool = False
