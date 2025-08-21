# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .adapters import BaseModelAdapter, initialize_model_adapter
    from .custom import CustomModel, DummyCustomModel, DummyT2IModel
    from .local_model import LocalChatModel, LocalImageModel, LocalModel, get_local_model
    from .model_apis import llm_ckpt, mockllm, openai_api

else:
    _import_structure = {
        'adapters': [
            'BaseModelAdapter',
            'initialize_model_adapter',
        ],
        'model_apis': [
            'openai_api',
            'mockllm',
            'llm_ckpt',
        ],
        'custom': [
            'CustomModel',
            'DummyCustomModel',
            'DummyT2IModel',
        ],
        'local_model': [
            'LocalModel',
            'get_local_model',
            'LocalChatModel',
            'LocalImageModel',
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
