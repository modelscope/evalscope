"""Registered runtimes for task-environment services."""

from .ms_enclave_docker import MsEnclaveDockerLease, MsEnclaveDockerRuntime

__all__ = [
    'MsEnclaveDockerLease',
    'MsEnclaveDockerRuntime',
]
