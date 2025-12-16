import logging
import time
from tqdm import tqdm
from typing import Optional


class TqdmLogging(tqdm):

    def __init__(self, *args, logger: Optional[logging.Logger] = None, log_interval: float = 30.0, **kwargs):
        """
        Args:
            logger: logging.Logger instance. If None, logging is disabled.
            log_interval: Interval in seconds to log progress. Default is 10 seconds.
            *args, **kwargs: Arguments passed to original tqdm.
        """
        super().__init__(*args, **kwargs)
        self.custom_logger = logger
        self.log_interval = log_interval
        self.last_log_time = time.time()

    def update(self, n=1):
        """Override update method to check if logging is needed."""
        super().update(n)

        if self.custom_logger and (time.time() - self.last_log_time >= self.log_interval):
            self._log_status()
            self.last_log_time = time.time()

    def close(self):
        """Override close method to ensure final log is printed."""
        if self.custom_logger:
            self._log_status(final=True)
        super().close()

    def _log_status(self, final: bool = False):
        """Core logic to generate and write log."""
        current_val = self.n
        total_val = self.total
        desc = self.desc or 'Task'

        percentage = (current_val / total_val) * 100 if total_val else 0.0

        elapsed_sec = time.time() - self.start_t

        # Calculate ETA
        eta_sec = 0
        if current_val > 0 and total_val:
            rate = current_val / elapsed_sec
            remaining = total_val - current_val
            eta_sec = remaining / rate

        elapsed_str = self.format_interval(elapsed_sec)
        eta_str = self.format_interval(eta_sec) if not final else '00:00'

        # Example: [Processing] 45.0%| 45/100 [Elapsed: 00:04, ETA: 00:05]
        log_msg = (
            f'[{desc}] '
            f'{percentage:.1f}%| '
            f'{current_val}/{total_val} '
            f'[Elapsed: {elapsed_str}, ETA: {eta_str}]'
        )

        if self.custom_logger:
            self.custom_logger.info(log_msg)
