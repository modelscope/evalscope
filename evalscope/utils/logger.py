import colorlog
import importlib.util as iutil
import logging
import os
from datetime import datetime
from logging import Logger
from typing import List, Optional

from evalscope.constants import BEIJING_TZ, USE_OSS, LoggingConstants

init_loggers = {}

# Use ReopenFileHandler on OSS/FUSE mounts so each record is visible immediately;
# fall back to the standard FileHandler on local filesystems.
# Beijing timezone (UTC+8) for log timestamps when USE_OSS=1


def beijing_converter(timestamp):
    """Convert a Unix timestamp to Beijing time (UTC+8) as a time.struct_time."""
    return datetime.fromtimestamp(timestamp, tz=BEIJING_TZ).timetuple()


if USE_OSS:

    class BeijingColoredFormatter(colorlog.ColoredFormatter):

        def converter(self, timestamp):
            return beijing_converter(timestamp)

    class BeijingPlainFormatter(logging.Formatter):

        def converter(self, timestamp):
            return beijing_converter(timestamp)

    ColoredFmtCls = BeijingColoredFormatter
    PlainFmtCls = BeijingPlainFormatter
else:
    ColoredFmtCls = colorlog.ColoredFormatter
    PlainFmtCls = logging.Formatter

color_detailed_formatter = ColoredFmtCls(LoggingConstants.COLOR_DETAILED_FORMAT, datefmt=LoggingConstants.DATE_FORMAT)
color_simple_formatter = ColoredFmtCls(LoggingConstants.COLOR_SIMPLE_FORMAT, datefmt=LoggingConstants.DATE_FORMAT)
plain_detailed_formatter = PlainFmtCls(LoggingConstants.DETAILED_FORMAT, datefmt=LoggingConstants.DATE_FORMAT)
plain_simple_formatter = PlainFmtCls(LoggingConstants.SIMPLE_FORMAT, datefmt=LoggingConstants.DATE_FORMAT)

DEFAULT_LEVEL = logging.DEBUG if os.getenv('EVALSCOPE_LOG_LEVEL', 'INFO') == 'DEBUG' else logging.INFO

logging.basicConfig(format=LoggingConstants.SIMPLE_FORMAT, level=logging.INFO, force=True)

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


class ReopenFileHandler(logging.FileHandler):
    """FileHandler that closes the file after every emit.

    On OSS/FUSE-mounted filesystems the FUSE driver only uploads data when
    the file descriptor is closed.  By reopening on each record this handler
    ensures every log line is visible on OSS in near real-time.

    Thread safety: emit() is invoked inside Handler.handle() which already
    holds self.lock, so no extra locking is needed here.
    """

    def __init__(self, filename: str, mode: str = 'a', encoding: str = 'utf-8'):
        # delay=True: skip opening the file in __init__; we open it ourselves in emit()
        super().__init__(filename, mode=mode, encoding=encoding, delay=True)
        self._first_write = True

    def emit(self, record: logging.LogRecord) -> None:
        """Open → format+write → flush → close for every log record."""
        # After the first write switch to append so we never truncate
        if not self._first_write:
            self.mode = 'a'
        self.stream = self._open()
        try:
            logging.StreamHandler.emit(self, record)
        finally:
            if self.stream is not None:
                try:
                    self.stream.flush()
                    self.stream.close()
                finally:
                    self.stream = None
            self._first_write = False


# Module-level handler class: resolved once at import time from the USE_OSS env var.
FILE_HANDLER_CLS = ReopenFileHandler if _USE_OSS else logging.FileHandler


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
        file_handler = FILE_HANDLER_CLS(log_file, mode=file_mode, encoding='utf-8')
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

    file_handler = FILE_HANDLER_CLS(target_path, mode=file_mode, encoding='utf-8')
    file_handler.setFormatter(plain_detailed_formatter if log_level == logging.DEBUG else plain_simple_formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)


def warn_once(logger: Logger, message: str) -> None:
    if message not in _warned:
        logger.warning(message)
        _warned.append(message)


_warned: List[str] = []
