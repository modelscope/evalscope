# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .base_adapter import BaseModelAdapter, initialize_model_adapter
    from .chat_adapter import ChatGenerationModelAdapter
    from .choice_adapter import ContinuationLogitsModelAdapter, MultiChoiceModelAdapter
    from .custom import CustomModel, DummyCustomModel
    from .custom_adapter import CustomModelAdapter
    from .local_model import LocalModel, get_local_model
    from .model import BaseModel, ChatBaseModel, OpenAIModel
    from .register import get_model_adapter
    from .server_adapter import ServerModelAdapter

else:
    _import_structure = {
        'base_adapter': [
            'BaseModelAdapter',
            'initialize_model_adapter',
        ],
        'chat_adapter': [
            'ChatGenerationModelAdapter',
        ],
        'choice_adapter': [
            'ContinuationLogitsModelAdapter',
            'MultiChoiceModelAdapter',
        ],
        'custom': [
            'CustomModel',
            'DummyCustomModel',
        ],
        'custom_adapter': [
            'CustomModelAdapter',
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
        'server_adapter': [
            'ServerModelAdapter',
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
