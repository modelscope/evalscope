import logging
import time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Optional

from .progress_tracker import ProgressTracker


class TqdmLogging(tqdm):
    """tqdm subclass with periodic log output.

    Adds two conveniences on top of standard tqdm:

    1. Periodic structured log output at a configurable interval so that
       progress is visible in log files even when the terminal bar is not
       rendered (e.g. in CI or remote SSH sessions).
    2. Automatic redirection of Python logging output through
       ``logging_redirect_tqdm`` so that log messages do not corrupt the
       progress bar display.
    """

    def __init__(
        self,
        *args,
        logger: Optional[logging.Logger] = None,
        log_interval: Optional[float] = 30.0,
        track_progress: bool = False,
        **kwargs,
    ):
        """
        Args:
            logger: logging.Logger instance. If None, logging is disabled.
            log_interval: Interval in seconds to log progress. Default is 30 seconds.
            track_progress: When True, each update() call is forwarded to the active
                ProgressTracker.  Set this only on the tqdm instance whose unit of
                work corresponds to a single tracked item (e.g. one eval sample or
                one perf request).  Outer wrapper bars should leave this False to
                avoid double-counting.
            *args, **kwargs: Arguments passed to original tqdm.
        """
        super().__init__(*args, **kwargs)
        self.custom_logger = logger
        self.log_interval = log_interval
        self.track_progress = track_progress
        self.last_log_time = time.time()
        self.last_log_n = -1

        # Resolve and cache the active ProgressTracker instance at construction time.
        # Using an instance variable avoids repeated class-level lookups and makes
        # the association explicit for the lifetime of this progress bar.
        self._progress_tracker: Optional[ProgressTracker] = None
        if self.track_progress:
            self._progress_tracker = ProgressTracker.get_current()

        # Initialize logging redirection to prevent logger from interrupting tqdm progress bar
        # If logger is None, it defaults to redirecting the root logger
        loggers = [self.custom_logger] if self.custom_logger else None
        self._redirect_tqdm = logging_redirect_tqdm(loggers=loggers)
        self._redirect_tqdm.__enter__()

        # Bulk-update tracker for already-completed items passed via `initial`
        if self.initial and self.initial > 0:
            self._update_tracker(self.initial)

    def _update_tracker(self, n: int) -> None:
        """Forward *n* completed units to the cached ProgressTracker, if any."""
        if self._progress_tracker is not None:
            self._progress_tracker.update(n)

    def update(self, n=1):
        """Override update method to check if logging is needed."""
        super().update(n)
        self._update_tracker(n)
        self.check_log()

    def set_description(self, desc=None, refresh=True):
        super().set_description(desc, refresh)

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
