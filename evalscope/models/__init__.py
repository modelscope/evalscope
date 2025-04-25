# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .adapters import (BaseModelAdapter, ChatGenerationModelAdapter, ContinuationLogitsModelAdapter,
                           CustomModelAdapter, MultiChoiceModelAdapter, ServerModelAdapter, T2IModelAdapter,
                           initialize_model_adapter)
    from .custom import CustomModel, DummyCustomModel
    from .local_model import LocalModel, get_local_model
    from .model import BaseModel, ChatBaseModel, OpenAIModel
    from .register import get_model_adapter

else:
    _import_structure = {
        'adapters': [
            'BaseModelAdapter',
            'initialize_model_adapter',
            'ChatGenerationModelAdapter',
            'ContinuationLogitsModelAdapter',
            'MultiChoiceModelAdapter',
            'CustomModelAdapter',
            'ServerModelAdapter',
            'T2IModelAdapter',
        ],
        'custom': [
            'CustomModel',
            'DummyCustomModel',
        ],
        'local_model': [
            'LocalModel',
            'get_local_model',
        ],
        'model': [
            'BaseModel',
            'ChatBaseModel',
            'OpenAIModel',
        ],
        'register': [
            'get_model_adapter',
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
