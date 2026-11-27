"""Run-to-run statistics for repeated performance measurements.

A load test reports point estimates; running the same configuration several
times and publishing only the mean hides how much the number moves between
runs (GPU clocks, cache warm-up, JIT, network noise). This module turns a list
of per-run metric values into a printable summary of the mean, sample standard
deviation, and a two-sided 95% confidence interval for the mean.

For percentile metrics such as P99 latency, the interval estimates the
uncertainty in the mean of the per-run percentiles.
"""

from dataclasses import dataclass
from math import sqrt
from typing import List, Optional, Sequence

__all__ = ['RunStatistics', 'summarize_runs', 't_critical']

#: Two-sided 95% critical values of Student's t, indexed by degrees of freedom
#: (``df = n - 1``). Embedded instead of adding scipy to the perf extra; numpy
#: has no t quantile function of its own. For ``df > 30`` the normal
#: approximation (1.96) is within ~4% and is used instead.
_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def t_critical(df: int) -> float:
    """Return the two-sided 95% Student's t critical value for ``df`` degrees of freedom."""
    if df < 1:
        raise ValueError(f'df must be >= 1, got {df}')
    return _T_CRITICAL_95.get(df, 1.96)


@dataclass(frozen=True)
class RunStatistics:
    """Summary of one metric across repeated runs of the same configuration."""

    n: int
    """Number of runs the summary is based on."""

    mean: float
    """Arithmetic mean of the per-run values."""

    std: Optional[float] = None
    """Sample standard deviation (``ddof=1``); ``None`` when ``n < 2``."""

    ci_low: Optional[float] = None
    """Lower bound of the two-sided 95% interval for the mean; ``None`` when ``n < 3``."""

    ci_high: Optional[float] = None
    """Upper bound of the two-sided 95% interval for the mean; ``None`` when ``n < 3``."""

    @property
    def ci_half_width(self) -> Optional[float]:
        """Half-width of the confidence interval, when defined."""
        if self.ci_low is None or self.ci_high is None:
            return None
        return (self.ci_high - self.ci_low) / 2

    def format(self, unit: str = '') -> str:
        """Render ``mean ± std (95% CI [low, high], n=N)``, degrading gracefully."""
        text = f'{self.mean:.4g}'
        if self.std is not None:
            text += f' ± {self.std:.4g}'
        if self.ci_low is not None and self.ci_high is not None:
            text += f' (95% CI [{self.ci_low:.4g}, {self.ci_high:.4g}], n={self.n})'
        else:
            text += f' (n={self.n})'
        return f'{text}{unit}'


def summarize_runs(values: Sequence[float]) -> Optional[RunStatistics]:
    """Summarize per-run metric values.

    Args:
        values: One finite metric value per completed run.

    Returns:
        ``None`` when there are no values; otherwise a :class:`RunStatistics`
        with ``std`` from ``n >= 2`` and the interval from ``n >= 3``.
    """
    clean: List[float] = [float(v) for v in values]
    n = len(clean)
    if n == 0:
        return None

    mean = sum(clean) / n
    if n < 2:
        return RunStatistics(n=n, mean=mean)

    variance = sum((v - mean) ** 2 for v in clean) / (n - 1)
    std = sqrt(variance)
    if n < 3:
        return RunStatistics(n=n, mean=mean, std=std)

    half_width = t_critical(n - 1) * std / sqrt(n)
    return RunStatistics(
        n=n,
        mean=mean,
        std=std,
        ci_low=mean - half_width,
        ci_high=mean + half_width,
    )
