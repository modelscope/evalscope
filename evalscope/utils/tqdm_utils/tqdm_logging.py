import logging
import re
import time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .progress_tracker import ProgressTracker


class TqdmLogging(tqdm):
    """tqdm subclass with periodic log output and optional hierarchical progress tracking.

    Progress tracking is opt-in: call ``TqdmLogging.set_tracker(tracker)`` once
    before starting a pipeline and every subsequent tqdm instance will
    automatically register/deregister itself with the tracker using its ``desc``
    string to derive the stage name and label.

    Standardised ``desc`` format (use consistently across the codebase)::

        'StageName[label]'   e.g.  'Evaluating[mmlu]', 'Predicting[mmlu@test]'

    When no label is present just use a plain string, e.g. ``'Processing'``.
    """

    # Class-level tracker shared across all instances in the same process.
    _tracker: Optional['ProgressTracker'] = None

    @classmethod
    def set_tracker(cls, tracker: Optional['ProgressTracker']) -> None:
        """Attach or detach a ProgressTracker.

        Call with a ``ProgressTracker`` instance before starting a pipeline and
        with ``None`` after it finishes to avoid leaking state across runs.
        """
        cls._tracker = tracker

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
        self._stage_depth: Optional[int] = None

        # Initialize logging redirection to prevent logger from interrupting tqdm progress bar
        # If logger is None, it defaults to redirecting the root logger
        loggers = [self.custom_logger] if self.custom_logger else None
        self._redirect_tqdm = logging_redirect_tqdm(loggers=loggers)
        self._redirect_tqdm.__enter__()

        # Register this tqdm instance with the tracker (if active)
        if TqdmLogging._tracker is not None:
            name, label = self._parse_desc(self.desc or '')
            self._stage_depth = TqdmLogging._tracker._enter_stage(name=name, total=self.total or 0, label=label)

    def update(self, n=1):
        """Override update method to advance tracker and check if logging is needed."""
        super().update(n)
        if TqdmLogging._tracker is not None and self._stage_depth is not None:
            TqdmLogging._tracker._update_stage(self._stage_depth, self.n)
        self.check_log()

    def set_description(self, desc=None, refresh=True):
        """Override to sync name/label updates (e.g. ``set_description('Evaluating[ceval]')``) to tracker."""
        super().set_description(desc, refresh)
        if TqdmLogging._tracker is not None and self._stage_depth is not None and desc:
            name, label = self._parse_desc(desc)
            TqdmLogging._tracker._update_stage(
                self._stage_depth,
                self.n,
                name=name or None,
                label=label or None,
            )

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

        # Mark this stage as completed in the tracker
        if TqdmLogging._tracker is not None and self._stage_depth is not None:
            TqdmLogging._tracker._exit_stage(self._stage_depth)
            self._stage_depth = None

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

    @staticmethod
    def _parse_desc(desc: str) -> tuple:
        """Extract ``(name, label)`` from a standardised tqdm desc string.

        Recognises ``'Name[label]'`` and ``'Name (label)'`` bracket formats.
        Falls back to ``(desc, '')`` when no brackets are present.

        Examples::

            'Predicting[mmlu@test]'  -> ('Predicting', 'mmlu@test')
            'Evaluating[mmlu]'       -> ('Evaluating', 'mmlu')
            'Scoring[batch]'         -> ('Scoring', 'batch')
            'Running[p4_n100]'       -> ('Running', 'p4_n100')
            'Processing'             -> ('Processing', '')
            'Generating datasets'    -> ('Generating datasets', '')
        """
        desc = desc.strip().rstrip(': ')
        m = re.match(r'^([^\[\(]+?)\s*[\[\(]([^\]\)]+)[\]\)]', desc)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return desc, ''
