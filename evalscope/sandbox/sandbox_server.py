import abc
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from .config import ContainerConfig


# Shared data models
class ExecuteCodeRequest(BaseModel):
    container_id: str
    code: str
    timeout: Optional[int] = None
    working_dir: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)


class ExecuteCommandRequest(BaseModel):
    container_id: str
    command: List[str]
    timeout: Optional[int] = None
    working_dir: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)


class FileOperationRequest(BaseModel):
    container_id: str
    path: str


class WriteFileRequest(FileOperationRequest):
    content: str
    binary: bool = False


class CodeOutput(BaseModel):
    output: Optional[Any] = None
    logs: str = ''
    error: Optional[str] = None
    status_code: int = 0


class ContainerInfo(BaseModel):
    container_id: str
    status: str
    image: str
    ports: Dict[str, str] = Field(default_factory=dict)


class SandboxServer(abc.ABC):
    """Abstract base class for sandbox servers."""

    def __init__(self):
        self.app = FastAPI(title='Sandbox API', lifespan=self.lifespan)
        self._setup_routes()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Application lifespan management."""
        # Startup
        await self.startup()
        yield
        # Shutdown
        await self.shutdown()

    def _setup_routes(self):
        """Setup FastAPI routes."""
        self.app.post('/container/create', response_model=ContainerInfo)(self.create_container)
        self.app.post('/container/delete')(self.delete_container)
        self.app.post('/execute/code', response_model=CodeOutput)(self.execute_code)
        self.app.post('/execute/command', response_model=CodeOutput)(self.execute_command)
        self.app.post('/file/read')(self.read_file)
        self.app.post('/file/write')(self.write_file)
        self.app.get('/')(self.hello)

    @abc.abstractmethod
    async def startup(self):
        """Initialize the sandbox server."""
        pass

    @abc.abstractmethod
    async def shutdown(self):
        """Cleanup the sandbox server."""
        pass

    @abc.abstractmethod
    async def create_container(self, config: ContainerConfig, background_tasks: BackgroundTasks) -> ContainerInfo:
        """Create a new container for sandbox execution."""
        pass

    @abc.abstractmethod
    async def delete_container(self, container_id: str) -> Dict[str, str]:
        """Delete a container."""
        pass

    @abc.abstractmethod
    async def execute_code(self, request: ExecuteCodeRequest) -> CodeOutput:
        """Execute code in a container."""
        pass

    @abc.abstractmethod
    async def execute_command(self, request: ExecuteCommandRequest) -> CodeOutput:
        """Execute a command in a container."""
        pass

    @abc.abstractmethod
    async def read_file(self, request: FileOperationRequest) -> Dict[str, Any]:
        """Read a file from a container."""
        pass

    @abc.abstractmethod
    async def write_file(self, request: WriteFileRequest) -> Dict[str, str]:
        """Write a file to a container."""
        pass

    async def hello(self) -> Dict[str, str]:
        """Root endpoint."""
        return {'message': 'Welcome to the Sandbox API'}

    def run(self, host: str = '0.0.0.0', port: int = 8000, **kwargs):
        """Run the server."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, **kwargs)
