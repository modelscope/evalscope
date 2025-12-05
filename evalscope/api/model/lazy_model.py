# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Lazy loading wrapper for Model to defer model initialization until first use.

This module provides LazyModel class that implements delayed model loading,
which is useful when predictions are cached and model loading can be avoided.
"""

from typing import TYPE_CHECKING, Any, Optional

from evalscope.utils.function_utils import thread_safe
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.model import Model
    from evalscope.config import TaskConfig

logger = get_logger()


class LazyModel:
    """
    Lazy loading wrapper for Model.

    This class defers the actual model loading until the first time any model
    method or attribute is accessed. This is particularly useful when using cached
    predictions, as it avoids unnecessary model loading overhead.

    The LazyModel acts as a transparent proxy to the real Model object, implementing
    the same interface through __getattr__ delegation.

    Example:
        >>> task_config = TaskConfig(model='Qwen/Qwen2.5-7B-Instruct', ...)
        >>> lazy_model = LazyModel(task_config)
        >>> # Model is NOT loaded yet
        >>>
        >>> # First access triggers model loading
        >>> output = lazy_model.generate(...)
        >>> # Model is now loaded and will be reused for subsequent calls
    """

    def __init__(self, task_config: 'TaskConfig'):
        """
        Initialize the lazy model wrapper.

        Args:
            task_config: Task configuration containing model parameters.
                Will be used to load the model when needed.
        """
        # Use object.__setattr__ to avoid triggering __setattr__
        object.__setattr__(self, '_task_config', task_config)
        object.__setattr__(self, '_model', None)
        object.__setattr__(self, '_is_loaded', False)

    @thread_safe
    def _ensure_loaded(self) -> 'Model':
        """
        Ensure the model is loaded, loading it if necessary.

        This method is called internally before delegating any attribute
        access to the real model.

        Returns:
            The loaded Model instance.
        """
        if not object.__getattribute__(self, '_is_loaded'):
            logger.info('Loading model for prediction...')

            from evalscope.api.model.model import get_model_with_task_config

            task_config = object.__getattribute__(self, '_task_config')
            model = get_model_with_task_config(task_config=task_config)

            object.__setattr__(self, '_model', model)
            object.__setattr__(self, '_is_loaded', True)

            logger.info('Model loaded successfully.')

        return object.__getattribute__(self, '_model')

    @property
    def is_loaded(self) -> bool:
        """
        Check if the model has been loaded.

        Returns:
            True if model is loaded, False otherwise.
        """
        return object.__getattribute__(self, '_is_loaded')

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying model.

        This method is called for any attribute that doesn't exist on LazyModel itself.
        It triggers model loading if not already loaded, then forwards the attribute
        access to the real model.

        Args:
            name: Name of the attribute to access.

        Returns:
            The attribute value from the underlying model.

        Raises:
            AttributeError: If the attribute doesn't exist on the model.
        """
        # Avoid infinite recursion for internal attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Ensure model is loaded, then delegate attribute access
        model = self._ensure_loaded()
        return getattr(model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Delegate attribute setting to the underlying model.

        Args:
            name: Name of the attribute to set.
            value: Value to set.
        """
        # Internal attributes are set directly
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            # Delegate to the real model
            model = self._ensure_loaded()
            setattr(model, name, value)

    def __repr__(self) -> str:
        """Return string representation of the LazyModel."""
        if object.__getattribute__(self, '_is_loaded'):
            model = object.__getattribute__(self, '_model')
            return f'LazyModel(loaded={True}, model={repr(model)})'
        else:
            task_config = object.__getattribute__(self, '_task_config')
            return f'LazyModel(loaded={False}, model_id={task_config.model_id})'

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        if object.__getattribute__(self, '_is_loaded'):
            model = object.__getattribute__(self, '_model')
            return str(model)
        else:
            task_config = object.__getattribute__(self, '_task_config')
            return f'LazyModel<{task_config.model_id}> (not loaded yet)'

    def __getstate__(self) -> dict:
        """
        Support for pickle serialization.

        Returns:
            Dictionary containing the state to serialize.
        """
        return {
            'task_config': object.__getattribute__(self, '_task_config'),
            'model': None,
            'is_loaded': False,
        }

    def __setstate__(self, state: dict) -> None:
        """
        Support for pickle deserialization.

        Args:
            state: Dictionary containing the state to restore.
        """
        object.__setattr__(self, '_task_config', state['task_config'])
        object.__setattr__(self, '_model', None)
        object.__setattr__(self, '_is_loaded', False)
