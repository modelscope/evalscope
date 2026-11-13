"""Confidence intervals for aggregated scores.

A single-run score is a point estimate: 87.5% over 20 samples and 87.5% over 1000 samples carry
different confidence. This module gives ratio metrics (every sample scored 0 or 1) an interval
estimate; continuous metrics need the percentile bootstrap planned as the second slice of #2011.

The interval describes item-sampling error only: "if we drew a different sample of equally
difficult items, how much would this score move". It does not capture decoding nondeterminism,
prompt layout sensitivity or judge variance.
"""

from math import sqrt
from statistics import NormalDist
from typing import Tuple

DEFAULT_CI_LEVEL = 0.95
"""Two-sided confidence level used for every interval."""

MIN_CONFIDENT_SAMPLES = 30
"""Below this many samples an interval is reported as low confidence."""

MAX_CONFIDENT_CI_WIDTH = 0.2
"""Intervals wider than this are reported as low confidence."""


def wilson_interval(successes: int, n: int, level: float = DEFAULT_CI_LEVEL) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and keeps a
    non-zero width when every sample succeeds or every sample fails, where the normal
    approximation collapses to a zero-width interval.

    Args:
        successes: Number of samples scored 1.
        n: Number of samples in the aggregation.
        level: Two-sided confidence level.

    Returns:
        ``(low, high)`` bounds of the interval, or ``(0.0, 0.0)`` when ``n`` is not positive.
    """
    if n <= 0:
        return (0.0, 0.0)

    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    # Floating-point residue can leave a bound a hair outside [0, 1]; clamp it so a metric
    # scored 0/0 keeps exact bounds instead of 2.8e-17.
    low = max(0.0, centre - margin)
    high = min(1.0, centre + margin)
    return (0.0 if low < 1e-12 else low, 1.0 if high > 1 - 1e-12 else high)


def ratio_ci_metadata(successes: int, n: int, level: float = DEFAULT_CI_LEVEL) -> dict:
    """Build the ``AggScore.metadata`` entry that describes a ratio metric's interval.

    Args:
        successes: Number of samples scored 1.
        n: Number of samples in the aggregation.
        level: Two-sided confidence level.

    Returns:
        Mapping with ``ci_low``, ``ci_high``, ``ci_level``, ``ci_method`` and
        ``low_confidence``; ``AggScore.metadata`` is the only extension point because
        ``AggScore`` forbids extra fields.
    """
    low, high = wilson_interval(successes, n, level)
    return {
        'ci_low': round(low, 6),
        'ci_high': round(high, 6),
        'ci_level': level,
        'ci_method': 'wilson',
        'low_confidence': n < MIN_CONFIDENT_SAMPLES or (high - low) > MAX_CONFIDENT_CI_WIDTH,
    }
