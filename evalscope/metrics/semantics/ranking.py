"""Direction-aware quality ratios, used to rank and filter results.

A raw metric value is not a ranking key. Sorting a list of runs by their reported numbers puts a
0-100 judge score above every ratio, and it puts a *high* word error rate above a low one. This
module normalizes a value into a "how good is it" ratio in ``[0, 1]``: bounded against the metric's
own ``value_range`` and inverted for a ``lower_is_better`` metric.

The result is a ranking key and a filter key only. It is never displayed and never persisted --
a synthesized 0-1 number would read like a score, and the whole point of the semantics contract is
that a reported score keeps its own scale and unit. The frontend primitive
``getBoundedQualityRatio`` in ``evalscope/web/src/domain/metric/metricFormat.ts`` applies the same
rule for colour scales.
"""

import math
from typing import Optional

from evalscope.api.metric.semantics import MetricDirection, MetricKind, MetricSemantics


def bounded_quality_ratio(value: Optional[float], semantics: Optional[MetricSemantics]) -> Optional[float]:
    """Normalize a value into a ``[0, 1]`` ratio where higher always means better.

    Args:
        value: Raw metric value in its native scale.
        semantics: Resolved semantics of the metric.

    Returns:
        The quality ratio, or ``None`` when the metric admits no such scale: a diagnostic, a
        metric without a ``value_range``, or one whose direction is ``none``. ``None`` means
        "not rankable", which callers must treat as unknown rather than as zero.
    """
    if value is None or semantics is None or not math.isfinite(float(value)):
        return None
    if semantics.kind is MetricKind.DIAGNOSTIC or semantics.direction is MetricDirection.NONE:
        return None
    value_range = semantics.value_range
    if value_range is None:
        return None

    # ValueRange already guarantees finite bounds with min < max, so the span is safe to divide by.
    span = value_range.max - value_range.min
    clamped = min(1.0, max(0.0, (float(value) - value_range.min) / span))
    return 1.0 - clamped if semantics.direction is MetricDirection.LOWER_IS_BETTER else clamped
