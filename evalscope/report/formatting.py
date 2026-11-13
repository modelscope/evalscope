# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared display formatting for report values.

This module is the single formatter for confidence intervals: the report
generator (which renders ``Subset.ci_display``) and the combinator (which
renders the CLI ``CI`` column) both call :func:`format_ci_value`, so the two
surfaces cannot drift.
"""

from typing import Any, Dict, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()


def format_ci_value(ci: Optional[Dict[str, Any]], n: Optional[int], semantics) -> str:
    """Render one subset's confidence interval, or ``''`` when there is none.

    Endpoints are formatted through the metric's own semantics (display
    multiplier, precision, unit) — the same treatment the score cell gets, so a
    0-5 quality score never renders as a percentage. The interval is asymmetric;
    it renders as ``[lo, hi]``, never ``±``. Every flag renders **alongside**
    the interval (never instead of it): ``low_confidence`` says the sample is
    small, ``wide_interval`` says the metric is high-variance.
    """
    if not isinstance(ci, dict):
        return ''
    ci_low, ci_high = ci.get('low'), ci.get('high')
    if ci_low is None or ci_high is None:
        return ''
    from evalscope.metrics.semantics.formatting import format_metric_value

    interval = f'[{format_metric_value(ci_low, semantics)}, {format_metric_value(ci_high, semantics)}]'
    notes = []
    if 'low_confidence' in (ci.get('flags') or []):
        notes.append(f'low confidence, n={n}')
    if 'wide_interval' in (ci.get('flags') or []):
        notes.append('wide interval')
    return f'{interval} ({"; ".join(notes)})' if notes else interval
