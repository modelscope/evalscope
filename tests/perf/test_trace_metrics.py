# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for TraceAccumulator + TraceLevelSummary (P3)."""
import unittest

from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.perf.utils.trace_metrics import TraceAccumulator, TraceLevelSummary


def _turn(
    *,
    trace_id: str,
    start: float,
    completed: float,
    ttft: float,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    success: bool = True,
    is_warmup: bool = False,
) -> BenchmarkData:
    d = BenchmarkData(success=success)
    d.is_warmup = is_warmup
    d.trace_id = trace_id
    d.start_time = start
    d.completed_time = completed
    d.first_chunk_latency = ttft
    d.query_latency = completed - start
    d.prompt_tokens = prompt_tokens
    d.completion_tokens = completion_tokens
    d.cached_tokens = cached_tokens
    return d


class TestTraceAccumulatorFiltering(unittest.TestCase):

    def test_skips_warmup_failed_and_single_turn_items(self):
        acc = TraceAccumulator()
        # Warmup is excluded.
        acc.feed(_turn(trace_id='t1', start=0, completed=1, ttft=0.1, prompt_tokens=10,
                       completion_tokens=5, is_warmup=True))
        # Failed is excluded.
        acc.feed(_turn(trace_id='t1', start=0, completed=1, ttft=0.1, prompt_tokens=10,
                       completion_tokens=5, success=False))
        # Single-turn (no trace_id) is excluded.
        single = _turn(trace_id='', start=0, completed=1, ttft=0.1, prompt_tokens=10, completion_tokens=5)
        single.trace_id = None
        acc.feed(single)

        self.assertEqual(acc.n_traces, 0)
        self.assertTrue(acc.to_summary().is_empty())


class TestTraceDerivedMetrics(unittest.TestCase):
    """Single-trace derivations: latency, TTFAT, decode TPS, cache hit, eligible."""

    def _three_turn_trace(self) -> TraceAccumulator:
        acc = TraceAccumulator()
        # Turn 1: start=0, done=2, ttft=0.5, prompt=100, completion=51, no cache.
        acc.feed(_turn(trace_id='t', start=0.0, completed=2.0, ttft=0.5,
                       prompt_tokens=100, completion_tokens=51, cached_tokens=0))
        # Turn 2: start=3, done=4.5, ttft=0.3, prompt=200, completion=21,
        # cache hit = 130 (server reports turn1's prompt+completion = 151 eligible).
        acc.feed(_turn(trace_id='t', start=3.0, completed=4.5, ttft=0.3,
                       prompt_tokens=200, completion_tokens=21, cached_tokens=130))
        # Turn 3: start=5, done=7, ttft=0.4, prompt=400, completion=31,
        # cache hit = 200 (eligible = 200+21 = 221).
        acc.feed(_turn(trace_id='t', start=5.0, completed=7.0, ttft=0.4,
                       prompt_tokens=400, completion_tokens=31, cached_tokens=200))
        return acc

    def test_latency_first_turn_ttft_ttfat(self):
        acc = self._three_turn_trace()
        state = acc._traces['t']  # internal access OK in same-package unit test
        self.assertAlmostEqual(state.latency, 7.0)  # last.completed - first.start
        self.assertAlmostEqual(state.first_turn_ttft, 0.5)
        # TTFAT = last.start + last.ttft - first.start = 5.0 + 0.4 - 0 = 5.4
        self.assertAlmostEqual(state.ttfat, 5.4)

    def test_decode_tps_arithmetic_mean(self):
        acc = self._three_turn_trace()
        state = acc._traces['t']
        # turn1: (51-1)/(2.0-0.5) = 33.333
        # turn2: (21-1)/(1.5-0.3) = 16.667
        # turn3: (31-1)/(2.0-0.4) = 18.75
        # mean = (33.333 + 16.667 + 18.75) / 3 ≈ 22.917
        self.assertAlmostEqual(state.decode_tps, (50/1.5 + 20/1.2 + 30/1.6) / 3, places=4)

    def test_cache_hit_rate_uses_total_prompt(self):
        acc = self._three_turn_trace()
        state = acc._traces['t']
        # sum(cached) = 0+130+200 = 330; sum(prompt) = 100+200+400 = 700
        self.assertAlmostEqual(state.cache_hit_rate, 330 / 700 * 100.0)

    def test_eligible_cache_hit_rate_skips_turn1_and_uses_prev_context(self):
        acc = self._three_turn_trace()
        state = acc._traces['t']
        # eligible: turn1=0, turn2=100+51=151, turn3=200+21=221
        # numerator counts only turns with eligible > 0: 130 + 200 = 330
        # denominator: 151 + 221 = 372
        self.assertAlmostEqual(state.eligible_cache_hit_rate, 330 / 372 * 100.0)


class TestTraceLevelSummaryAggregation(unittest.TestCase):

    def test_summary_aggregates_across_traces(self):
        acc = TraceAccumulator()
        # Trace A: single turn, latency=1.0, ttft=0.1, completion=11, no cache.
        acc.feed(_turn(trace_id='a', start=0.0, completed=1.0, ttft=0.1,
                       prompt_tokens=50, completion_tokens=11))
        # Trace B: single turn, latency=3.0, ttft=0.5, completion=31, no cache.
        acc.feed(_turn(trace_id='b', start=0.0, completed=3.0, ttft=0.5,
                       prompt_tokens=50, completion_tokens=31))

        summary = acc.to_summary()
        self.assertEqual(summary.n_traces, 2)
        # Six metric rows.
        metric_names = [r.metric for r in summary.rows]
        self.assertIn('Latency (s)', metric_names)
        self.assertIn('TTFAT (s)', metric_names)
        self.assertIn('First-Turn TTFT (s)', metric_names)
        self.assertIn('Decode TPS', metric_names)
        self.assertIn('Cache Hit Rate (%)', metric_names)
        self.assertIn('Eligible Cache Hit Rate (%)', metric_names)

        latency_row = next(r for r in summary.rows if r.metric == 'Latency (s)')
        self.assertAlmostEqual(latency_row.min, 1.0)
        self.assertAlmostEqual(latency_row.max, 3.0)
        self.assertAlmostEqual(latency_row.mean, 2.0)

    def test_table_render_includes_columns(self):
        acc = TraceAccumulator()
        acc.feed(_turn(trace_id='a', start=0.0, completed=1.0, ttft=0.1,
                       prompt_tokens=10, completion_tokens=5))
        table = acc.to_summary().to_table()
        for col in ('mean', 'min', 'p50', 'p90', 'p95', 'p99', 'max'):
            self.assertIn(col, table)
        self.assertIn('Latency (s)', table)

    def test_empty_summary_is_empty_returns_blank_table(self):
        summary = TraceLevelSummary()
        self.assertTrue(summary.is_empty())
        self.assertEqual(summary.to_table(), '')

    def test_to_dict_roundtrip_shape(self):
        acc = TraceAccumulator()
        acc.feed(_turn(trace_id='a', start=0.0, completed=1.0, ttft=0.1,
                       prompt_tokens=10, completion_tokens=5))
        d = acc.to_summary().to_dict()
        self.assertEqual(d['n_traces'], 1)
        self.assertEqual(len(d['rows']), 6)
        # Every row carries the seven aggregate keys.
        for row in d['rows']:
            self.assertIn('metric', row)
            for stat in ('mean', 'min', 'p50', 'p90', 'p95', 'p99', 'max'):
                self.assertIn(stat, row)


if __name__ == '__main__':
    unittest.main(buffer=False)
