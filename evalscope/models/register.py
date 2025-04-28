from evalscope.constants import OutputType
from .adapters import *

MODEL_ADAPTERS = {}


def register_model_adapter(name):
    """
    Decorator to register a model adapter with a given name.
    :param name: The name of the model adapter.
    """

    def decorator(adapter):
        if name in MODEL_ADAPTERS:
            raise ValueError(f"Model adapter '{name}' is already registered.")
        MODEL_ADAPTERS[name] = adapter
        return adapter

    return decorator


def get_model_adapter(name):
    """
    Retrieve a registered model adapter by name.
    :param name: The name of the model adapter.
    :return: The model adapter class or function.
    """
    if name not in MODEL_ADAPTERS:
        raise ValueError(
            f"Model adapter '{name}' is not registered. Available model adapters: {list(MODEL_ADAPTERS.keys())}")
    return MODEL_ADAPTERS[name]


def register_model_adapter_class(cls, name=None):
    """
    Register a model adapter class.
    :param cls: The model adapter class to register
    :param name: Optional name for the model adapter. If not provided, the class name will be used.
    """
    if name is None:
        name = cls.__name__
    if name in MODEL_ADAPTERS:
        raise ValueError(f"Model adapter class '{name}' is already registered.")
    MODEL_ADAPTERS[name] = cls


# register all model adapters
register_model_adapter_class(BaseModelAdapter, name='base')
register_model_adapter_class(ChatGenerationModelAdapter, name=OutputType.GENERATION)
register_model_adapter_class(ContinuationLogitsModelAdapter, name=OutputType.LOGITS)
register_model_adapter_class(MultiChoiceModelAdapter, name=OutputType.MULTIPLE_CHOICE)
register_model_adapter_class(CustomModelAdapter, name='custom')
register_model_adapter_class(ServerModelAdapter, name='server')
register_model_adapter_class(T2IModelAdapter, name=OutputType.IMAGE_GENERATION)
