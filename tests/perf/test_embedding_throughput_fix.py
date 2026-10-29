# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit test for embedding input token throughput calculation.

Verifies that avg_input_token_throughput uses wall_time as the denominator
(not total_first_chunk_latency, which is the sum of per-request latencies
and grows with concurrency).
"""
import unittest

from evalscope.perf.utils.benchmark_util import MetricsAccumulator


class TestEmbeddingInputThroughput(unittest.TestCase):
    """Test that embedding input throughput scales correctly with concurrency."""

    def _build_accumulator(self, n_success, latency, prompt_tokens, wall_time):
        """Build a MetricsAccumulator with pre-set internal state.

        Simulates n_success embedding requests, each with the given latency
        and prompt_tokens, all finishing within wall_time seconds.
        """
        acc = MetricsAccumulator(concurrency=1, rate=-1)
        acc.n_success = n_success
        acc.total_latency = n_success * latency
        acc.total_first_chunk_latency = n_success * latency  # non-stream: == query_latency
        acc.total_prompt_tokens = n_success * prompt_tokens
        acc.total_completion_tokens = 0
        # Set wall_time window
        acc._wall_start = 0.0
        acc._wall_end = wall_time
        return acc

    def test_throughput_scales_with_concurrency(self):
        """Input throughput should scale with concurrency, not stay flat.

        Before the fix, throughput = total_prompt_tokens / total_first_chunk_latency
        = (N * tokens) / (N * latency) = tokens / latency ? flat regardless of N.

        After the fix, throughput = total_prompt_tokens / wall_time
        = (N * tokens) / wall_time ? scales with N.
        """
        latency = 0.05
        prompt_tokens = 128

        # 1 request, wall_time = latency
        acc_1 = self._build_accumulator(n_success=1, latency=latency, prompt_tokens=prompt_tokens, wall_time=latency)
        result_1 = acc_1.to_result()
        tp_1 = result_1.avg_input_token_throughput

        # 10 requests, wall_time = latency (all concurrent)
        acc_10 = self._build_accumulator(n_success=10, latency=latency, prompt_tokens=prompt_tokens, wall_time=latency)
        result_10 = acc_10.to_result()
        tp_10 = result_10.avg_input_token_throughput

        # With fix: tp_10 should be ~10x tp_1
        # Without fix: tp_10 == tp_1 (flat)
        self.assertGreater(
            tp_10, tp_1 * 5,
            msg=f'Throughput should scale with concurrency: '
            f'1 req -> {tp_1:.1f} tok/s, 10 req -> {tp_10:.1f} tok/s. '
            f'Expected at least 5x.'
        )

    def test_throughput_matches_wall_time_formula(self):
        """Input throughput should equal total_prompt_tokens / wall_time."""
        n = 10
        latency = 0.05
        prompt_tokens = 128
        wall_time = 0.05  # all concurrent

        acc = self._build_accumulator(n_success=n, latency=latency, prompt_tokens=prompt_tokens, wall_time=wall_time)
        result = acc.to_result()

        expected = (n * prompt_tokens) / wall_time
        actual = result.avg_input_token_throughput

        self.assertAlmostEqual(
            actual, expected, delta=expected * 0.01,
            msg=f'Should be {expected:.1f} (total_tokens/wall_time), got {actual:.1f}'
        )

    def test_old_formula_would_be_wrong(self):
        """Verify the old formula (total_first_chunk_latency denominator) would be wrong."""
        n = 10
        latency = 0.05
        prompt_tokens = 128
        wall_time = 0.05

        acc = self._build_accumulator(n_success=n, latency=latency, prompt_tokens=prompt_tokens, wall_time=wall_time)
        result = acc.to_result()

        # Old (buggy): total_prompt_tokens / total_first_chunk_latency = tokens / latency
        old_expected = prompt_tokens / latency
        actual = result.avg_input_token_throughput

        # Fixed value should be 10x the old buggy value
        self.assertNotAlmostEqual(
            actual, old_expected, delta=old_expected * 0.01,
            msg=f'Fixed ({actual:.1f}) should NOT match old formula ({old_expected:.1f}). '
            f'Old formula gives flat value regardless of concurrency.'
        )


if __name__ == '__main__':
    unittest.main()
