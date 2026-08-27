"""Unit tests for the nearest-rank percentile calculation in perf metrics.

``calculate_percentiles`` used ``int(n * p / 100)`` as the index, which biased
every percentile one rank high (n=100 -> p99 = max; n=2 -> p50 = larger value).
These tests pin the nearest-rank semantics: the p-th percentile is the value at
1-based rank ceil(p / 100 * n).
"""
import math

import pytest

from evalscope.perf.utils.db_util import calculate_percentiles


class TestCalculatePercentiles:

    def test_nearest_rank_on_one_hundred_values(self):
        # Previously returned {50: 50, 99: 99} (one rank too high).
        assert calculate_percentiles(list(range(100)), [50, 99]) == {50: 49, 99: 98}

    def test_median_of_two_values_is_the_smaller(self):
        # int(2 * 50 / 100) == 1 picked the larger value; nearest rank picks rank 1.
        assert calculate_percentiles([1.0, 2.0], [50]) == {50: 1.0}

    def test_single_value_all_percentiles(self):
        assert calculate_percentiles([7.0], [0, 1, 50, 99, 100]) == {0: 7.0, 1: 7.0, 50: 7.0, 99: 7.0, 100: 7.0}

    def test_percentile_zero_is_min(self):
        assert calculate_percentiles([3.0, 1.0, 2.0], [0]) == {0: 1.0}

    def test_percentile_hundred_or_more_is_max(self):
        assert calculate_percentiles([3.0, 1.0, 2.0], [100, 150]) == {100: 3.0, 150: 3.0}

    def test_empty_data_returns_nan(self):
        result = calculate_percentiles([], [0, 50, 100])
        assert all(math.isnan(v) for v in result.values())

    def test_missing_value_returns_nan(self):
        result = calculate_percentiles([None], [0, 50, 100])
        assert all(math.isnan(v) for v in result.values())

    def test_input_list_is_sorted_in_place(self):
        data = [9.0, 1.0, 5.0]
        calculate_percentiles(data, [50])
        assert data == [1.0, 5.0, 9.0]

    def test_small_sample_nearest_rank_boundaries(self):
        # n=3: p1 -> ceil(0.03)=1 -> idx 0; p50 -> ceil(1.5)=2 -> idx 1; p99 -> ceil(2.97)=3 -> idx 2.
        assert calculate_percentiles([10.0, 20.0, 30.0], [1, 50, 99]) == {1: 10.0, 50: 20.0, 99: 30.0}

    @pytest.mark.parametrize('n', [1, 2, 3, 7, 10, 100])
    def test_percentiles_never_exceed_max_or_fall_below_min(self, n):
        data = list(range(n))
        result = calculate_percentiles(data, [1, 5, 25, 50, 75, 95, 99])
        assert all(0 <= v <= n - 1 for v in result.values())
