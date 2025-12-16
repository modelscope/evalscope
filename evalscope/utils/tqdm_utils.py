import logging
import time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Optional


class TqdmLogging(tqdm):

    def __init__(self, *args, logger: Optional[logging.Logger] = None, log_interval: Optional[float] = 30.0, **kwargs):
        """
        Args:
            logger: logging.Logger instance. If None, logging is disabled.
            log_interval: Interval in seconds to log progress. Default is 30 seconds.
            *args, **kwargs: Arguments passed to original tqdm.
        """
        super().__init__(*args, **kwargs)
        self.custom_logger = logger
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.last_log_n = -1

        # Initialize logging redirection to prevent logger from interrupting tqdm progress bar
        # If logger is None, it defaults to redirecting the root logger
        loggers = [self.custom_logger] if self.custom_logger else None
        self._redirect_tqdm = logging_redirect_tqdm(loggers=loggers)
        self._redirect_tqdm.__enter__()

    def update(self, n=1):
        """Override update method to check if logging is needed."""
        super().update(n)
        self.check_log()

    def check_log(self):
        """Check if logging is needed based on time interval."""
        if self.custom_logger and self.log_interval is not None and (
            time.time() - self.last_log_time >= self.log_interval
        ):
            self._log_status()
            self.last_log_time = time.time()

    def close(self):
        """Override close method to ensure final log is printed and clean up redirection."""
        # Only log if the current progress (n) hasn't been logged yet
        if self.custom_logger and self.n != self.last_log_n:
            self._log_status()

        # Exit logging redirection
        if hasattr(self, '_redirect_tqdm'):
            self._redirect_tqdm.__exit__(None, None, None)
            del self._redirect_tqdm

        super().close()

    def _log_status(self):
        """
        Generate log using tqdm native calculation logic.
        """
        # 1. Get current status dictionary from tqdm
        # Contains: n, total, elapsed, rate (smoothed), unit, etc.
        d = self.format_dict.copy()

        # 2. Define log-specific format string
        # Variables like {desc}, {percentage}, {remaining} are standard placeholders supported by format_meter
        # We removed {bar} to keep only text information
        # {remaining} is the ETA calculated by tqdm
        log_fmt = '{desc} {percentage:3.0f}%| {n_fmt}/{total_fmt} [Elapsed: {elapsed} < Remaining: {remaining}, {rate_fmt}]'  # noqa E501

        # 3. Force override bar_format
        d['bar_format'] = log_fmt

        # 4. Call tqdm static method to render
        # format_meter will use rate and total in d to calculate remaining automatically
        log_msg = tqdm.format_meter(**d)

        if self.custom_logger:
            self.custom_logger.info(log_msg)
            self.last_log_n = self.n
