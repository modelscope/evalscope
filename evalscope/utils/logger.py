import importlib.util as iutil
import logging
import os
from typing import Optional

init_loggers = {}

detailed_format = '%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s'
simple_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

detailed_formatter = logging.Formatter(detailed_format)
simple_formatter = logging.Formatter(simple_format)
DEFAULT_LEVEL = logging.DEBUG if os.getenv('EVALSCOPE_LOG_LEVEL', 'INFO') == 'DEBUG' else logging.INFO

logging.basicConfig(format=simple_format, level=DEFAULT_LEVEL, force=True)

# set logging level
logging.getLogger('datasets').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('modelscope').setLevel(logging.ERROR)


def get_logger(log_file: Optional[str] = None, log_level: int = DEFAULT_LEVEL, file_mode: str = 'w', force=False):
    """Get logging logger

    Args:
        log_file: Log filename, if specified, file handler will be added to
            logger
        log_level: Logging level.
        file_mode: Specifies the mode to open the file, if filename is
            specified (if filemode is unspecified, it defaults to 'w').
    """

    logger_name = __name__.split('.')[0]
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    if logger_name in init_loggers:
        if force:
            logger.setLevel(log_level)
            for handler in logger.handlers:
                handler.setLevel(log_level)
                handler.setFormatter(detailed_formatter if log_level == logging.DEBUG else simple_formatter)
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
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    for handler in handlers:
        handler.setFormatter(detailed_formatter if log_level == logging.DEBUG else simple_formatter)
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


def add_file_handler_if_needed(logger, log_file, file_mode, log_level):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return

    if iutil.find_spec('torch') is not None:
        from modelscope.utils.torch_utils import is_master

        is_worker0 = is_master()
    else:
        is_worker0 = True

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(detailed_formatter if log_level == logging.DEBUG else simple_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
