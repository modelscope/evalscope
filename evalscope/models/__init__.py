# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.models.base_adapter import BaseModelAdapter, initialize_model_adapter
from evalscope.models.chat_adapter import ChatGenerationModelAdapter
from evalscope.models.choice_adapter import ContinuationLogitsModelAdapter, MultiChoiceModelAdapter
from evalscope.models.custom import CustomModel
from evalscope.models.custom_adapter import CustomModelAdapter
from evalscope.models.local_model import LocalModel, get_local_model
from evalscope.models.model import BaseModel, ChatBaseModel, OpenAIModel
from evalscope.models.register import get_model_adapter
from evalscope.models.server_adapter import ServerModelAdapter

__all__ = [
    'CustomModel', 'BaseModel', 'ChatBaseModel', 'OpenAIModel', 'BaseModelAdapter', 'ChatGenerationModelAdapter',
    'MultiChoiceModelAdapter', 'ContinuationLogitsModelAdapter', 'CustomModelAdapter', 'ServerModelAdapter',
    'LocalModel', 'get_local_model', 'initialize_model_adapter', 'get_model_adapter'
]
