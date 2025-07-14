# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .config import ContainerConfig, DockerContainerConfig
    from .docker import DockerSandboxClient, DockerSandboxServer
    from .sandbox_client import SandboxClient
    from .sandbox_server import SandboxServer
else:
    _import_structure = {
        'config': [
            'ContainerConfig',
            'DockerContainerConfig',
        ],
        'docker': [
            'DockerSandboxClient',
            'DockerSandboxServer',
        ],
        'sandbox_client': [
            'SandboxClient',
        ],
        'sandbox_server': [
            'SandboxServer',
        ],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
