# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .app import create_app
    from .arguments import add_argument

else:
    _import_structure = {
        'app': [
            'create_app',
        ],
        'arguments': [
            'add_argument',
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
