import colorlog
import importlib.util as iutil
import logging
import os
from logging import Logger
from typing import List, Optional

init_loggers = {}
# Define log formats
data_format = '%Y-%m-%d %H:%M:%S'
# For console output
color_detailed_format = '%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d - %(log_color)s%(levelname)s%(reset)s: %(message)s'  # noqa:E501
color_simple_format = '%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s: %(message)s'
color_detailed_formatter = colorlog.ColoredFormatter(color_detailed_format, datefmt=data_format)
color_simple_formatter = colorlog.ColoredFormatter(color_simple_format, datefmt=data_format)
# For file output
detailed_format = '%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s: %(message)s'  # noqa:E501
simple_format = '%(asctime)s - %(name)s - %(levelname)s: %(message)s'
plain_detailed_formatter = logging.Formatter(detailed_format, datefmt=data_format)
plain_simple_formatter = logging.Formatter(simple_format, datefmt=data_format)

DEFAULT_LEVEL = logging.DEBUG if os.getenv('EVALSCOPE_LOG_LEVEL', 'INFO') == 'DEBUG' else logging.INFO

logging.basicConfig(format=simple_format, level=logging.INFO, force=True)

# set logging level
logging.getLogger('datasets').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('modelscope').setLevel(logging.ERROR)

info_set = set()
warning_set = set()


def info_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in info_set:
        return
    info_set.add(hash_id)
    self.info(msg)


def warning_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in warning_set:
        return
    warning_set.add(hash_id)
    self.warning(msg)


def get_logger(
    log_file: Optional[str] = None,
    name: Optional[str] = None,
    log_level: int = DEFAULT_LEVEL,
    file_mode: str = 'w',
    force: bool = False,
):
    """Get logging logger

    Args:
        log_file: Log filename. If specified, a file handler will be added to the logger.
        name: Logical component name. Used to derive the logger name.
        log_level: Logging level to set.
        file_mode: Mode to open the file when log_file is provided (default 'w').
        force: If True, reconfigure the existing logger (levels, formatters, handlers).
    """

    if name:
        logger_name = f"evalscope.{name.split('.')[-1]}"
    else:
        logger_name = 'evalscope'
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    if logger_name in init_loggers:
        if force:
            logger.setLevel(log_level)
            for handler in logger.handlers:
                handler.setLevel(log_level)
                # Select formatter by handler type
                if isinstance(handler, logging.FileHandler):
                    handler.setFormatter(
                        plain_detailed_formatter if log_level == logging.DEBUG else plain_simple_formatter
                    )
                else:
                    handler.setFormatter(
                        color_detailed_formatter if log_level == logging.DEBUG else color_simple_formatter
                    )
            # Ensure file handler points to current log_file (replace if needed)
            add_file_handler_if_needed(logger, log_file, file_mode, log_level)
        return logger

    # handle duplicate logs to the console
    torch_dist = False
    is_worker0 = True
    if iutil.find_spec('torch') is not None:
        from modelscope.utils.torch_utils import is_dist, is_master

        torch_dist = is_dist()
        is_worker0 = is_master()

    if torch_dist:
        for handler in logger.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode, encoding='utf-8')
        handlers.append(file_handler)

    for handler in handlers:
        # 区分不同类型的 handler，使用相应的格式化器
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(plain_detailed_formatter if log_level == logging.DEBUG else plain_simple_formatter)
        else:
            handler.setFormatter(color_detailed_formatter if log_level == logging.DEBUG else color_simple_formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if is_worker0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    init_loggers[logger_name] = True

    return logger


def configure_logging(debug: bool, log_file: Optional[str] = None):
    """Configure logging level based on the debug flag."""
    if log_file:
        get_logger(log_file=log_file, force=True)
    if debug:
        get_logger(log_level=logging.DEBUG, force=True)


def add_file_handler_if_needed(
    logger: logging.Logger,
    log_file: Optional[str],
    file_mode: str,
    log_level: int,
) -> None:
    """Ensure logger has a FileHandler targeting log_file.
    - If no FileHandler exists, add one.
    - If a FileHandler exists but points to a different file, replace it.
    """
    if log_file is None:
        return

    # Only worker-0 writes files
    if iutil.find_spec('torch') is not None:
        from modelscope.utils.torch_utils import is_master
        is_worker0 = is_master()
    else:
        is_worker0 = True

    if not is_worker0:
        return

    target_path = os.path.abspath(log_file)
    existing_file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]

    # If there is a FileHandler already pointing to the target file, nothing to do.
    for fh in existing_file_handlers:
        try:
            if os.path.abspath(getattr(fh, 'baseFilename', '')) == target_path:
                return
        except Exception:
            # If any issue retrieving baseFilename, fall through to replacement
            pass

    # Replace all existing FileHandlers with the new one
    for fh in existing_file_handlers:
        try:
            logger.removeHandler(fh)
            fh.flush()
            fh.close()
        except Exception:
            pass

    file_handler = logging.FileHandler(target_path, file_mode, encoding='utf-8')
    file_handler.setFormatter(plain_detailed_formatter if log_level == logging.DEBUG else plain_simple_formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)


def warn_once(logger: Logger, message: str) -> None:
    if message not in _warned:
        logger.warning(message)
        _warned.append(message)


_warned: List[str] = []
