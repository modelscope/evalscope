import functools
import inspect
from typing import Callable, Optional

from .logger import get_logger

logger = get_logger()


def deprecated(since: str, remove_in: Optional[str] = None, alternative: Optional[str] = None) -> Callable:
    """
    Decorator to mark functions as deprecated.

    :param since: String indicating the version since deprecation
    :param remove_in: Optional string indicating the version when it will be removed
    :param alternative: Optional string suggesting an alternative
    :return: Decorated function
    """

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the file name where the function is defined
            file_name = inspect.getfile(func)

            # Construct the warning message
            warning_parts = [
                f'{func.__name__} in {file_name} has been deprecated since version {since}',
                f'and will be removed in version {remove_in}' if remove_in else None,
                f'Use {alternative} instead' if alternative else None
            ]
            warning_message = '. '.join(filter(None, warning_parts))

            # Log the warning
            logger.warning(warning_message)

            return func(*args, **kwargs)

        return wrapper

    return decorator
