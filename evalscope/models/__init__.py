# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .model_apis import llm_ckpt, mockllm, openai_api

else:
    _import_structure = {
        'model_apis': [
            'openai_api',
            'mockllm',
            'llm_ckpt',
        ]
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
