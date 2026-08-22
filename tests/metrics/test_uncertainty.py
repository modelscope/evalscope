"""Unit tests for the fixed-item statistics helpers in evalscope.metrics.

These lock in the Wilson score interval and the exact paired McNemar test
used for comparing models on the same fixed item set.
"""

from evalscope.metrics.utils.functions import paired_mcnemar, wilson_ci


class TestWilsonCi:

    def test_center_is_p(self) -> None:
        lo, hi = wilson_ci(50, 100)
        assert abs((lo + hi) / 2 - 0.5) < 1e-9
        assert lo < 0.5 < hi

    def test_extreme_rates_stay_in_unit_interval(self) -> None:
        lo, hi = wilson_ci(0, 10)
        assert lo == 0.0
        lo, hi = wilson_ci(10, 10)
        assert hi == 1.0

    def test_reference_interval(self) -> None:
        # Wilson 95% interval for 50/100 is (0.404, 0.596) to 3 decimals.
        lo, hi = wilson_ci(50, 100)
        assert abs(lo - 0.404) < 0.001
        assert abs(hi - 0.596) < 0.001


class TestPairedMcNemar:

    def test_classic_discordant_pair(self) -> None:
        # The textbook (1, 9) discordant pair yields a two-sided p of ~0.0215.
        p = paired_mcnemar(1, 9)
        assert abs(p - 0.02148) < 0.0001

    def test_no_discordance(self) -> None:
        assert paired_mcnemar(0, 0) == 1.0

    def test_one_sided(self) -> None:
        assert paired_mcnemar(1, 9, two_sided=False) < 0.011
