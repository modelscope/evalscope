"""Tests for direction-aware quality ratios."""

import math
import pytest

from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
from evalscope.metrics.semantics.ranking import bounded_quality_ratio


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_not_rankable(value: float) -> None:
    semantics = SEMANTIC_BASELINES['quality.error_rate.ratio']

    assert bounded_quality_ratio(value, semantics) is None
