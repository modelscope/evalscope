"""Registered runtimes for task-environment services."""

from .ms_enclave_docker import MsEnclaveDockerLease, MsEnclaveDockerRuntime
from .remote import RemoteEnvironmentLease, RemoteEnvironmentRuntime

__all__ = [
    'MsEnclaveDockerLease',
    'MsEnclaveDockerRuntime',
    'RemoteEnvironmentLease',
    'RemoteEnvironmentRuntime',
]
