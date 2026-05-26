# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for WorkloadTimeline + WorkloadThroughput (P4)."""
import unittest

from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.perf.utils.workload_timeline import WorkloadThroughput, WorkloadTimeline


def _ok(*, start: float, completed: float, prompt: int, completion: int, cached: int = 0,
        is_warmup: bool = False, success: bool = True) -> BenchmarkData:
    d = BenchmarkData(success=success)
    d.is_warmup = is_warmup
    d.start_time = start
    d.completed_time = completed
    d.prompt_tokens = prompt
    d.completion_tokens = completion
    d.cached_tokens = cached
    return d


class TestWorkloadTimelineFeed(unittest.TestCase):

    def test_skips_warmup_failed_and_zero_timestamps(self):
        tl = WorkloadTimeline()
        tl.feed(_ok(start=1.0, completed=2.0, prompt=10, completion=5, is_warmup=True))
        tl.feed(_ok(start=1.0, completed=2.0, prompt=10, completion=5, success=False))
        tl.feed(_ok(start=0.0, completed=0.0, prompt=10, completion=5))  # zero timestamps
        self.assertEqual(tl.n_points, 0)
        self.assertEqual(tl.wall_time, 0.0)

    def test_cumulative_counts_and_wall_time(self):
        tl = WorkloadTimeline()
        tl.feed(_ok(start=100.0, completed=101.0, prompt=20, completion=10, cached=0))
        tl.feed(_ok(start=100.5, completed=102.0, prompt=30, completion=15, cached=20))
        tl.feed(_ok(start=101.0, completed=104.0, prompt=40, completion=20, cached=35))
        # Cumulative: completion = 10 + 15 + 20 = 45
        # new_prompt = 20 + (30-20) + (40-35) = 35
        # cached_prompt = 0 + 20 + 35 = 55
        # total_prompt = new + cached = 90 = 20 + 30 + 40 ✓
        self.assertEqual(tl.n_points, 3)
        last_overall = tl.overall_rates()
        # wall_time = 104 - 100 = 4
        self.assertAlmostEqual(tl.wall_time, 4.0)
        # total / wall = 90 / 4 = 22.5
        self.assertAlmostEqual(last_overall[0], 22.5)
        # new / wall = 35 / 4 = 8.75
        self.assertAlmostEqual(last_overall[1], 8.75)
        # cached / wall = 55 / 4 = 13.75
        self.assertAlmostEqual(last_overall[2], 13.75)
        # completion / wall = 45 / 4 = 11.25
        self.assertAlmostEqual(last_overall[3], 11.25)

    def test_wall_start_locked_on_first_feed(self):
        """A later request with a smaller start_time must not retroactively
        shift wall_start; otherwise already-appended ``t`` offsets would
        silently become wrong (they are never rewritten).
        """
        tl = WorkloadTimeline()
        tl.feed(_ok(start=100.0, completed=101.0, prompt=10, completion=5))
        # Second request started *earlier* but completed later (worker race).
        tl.feed(_ok(start=99.0, completed=102.0, prompt=10, completion=5))
        # wall_start stayed at 100; second point's t = 102 - 100 = 2.0.
        # Old "min" logic would have made wall_start=99 and silently broken
        # the first point's already-stored t (left at 1.0 instead of 2.0).
        self.assertEqual(tl._wall_start, 100.0)
        self.assertAlmostEqual(tl._points[0].t, 1.0)
        self.assertAlmostEqual(tl._points[1].t, 2.0)

    def test_new_prompt_clamped_when_cached_exceeds_prompt(self):
        tl = WorkloadTimeline()
        # Pathological: server reports cached > prompt (chat-template inflation).
        # WorkloadTimeline must clamp new_prompt to >= 0 to keep rates non-negative.
        tl.feed(_ok(start=1.0, completed=2.0, prompt=10, completion=5, cached=50))
        rates = tl.overall_rates()
        self.assertGreaterEqual(rates[1], 0.0)  # new_prompt non-negative


class TestSteadyStateAndLastWindow(unittest.TestCase):

    def test_last_window_falls_back_to_overall_when_run_shorter(self):
        tl = WorkloadTimeline()
        tl.feed(_ok(start=100.0, completed=101.0, prompt=20, completion=10))
        tl.feed(_ok(start=100.5, completed=105.0, prompt=20, completion=10))
        # wall = 5s < 30s; last_window_rates(30) should equal overall_rates
        self.assertEqual(tl.last_window_rates(30.0), tl.overall_rates())

    def test_last_window_uses_anchor_inside_run(self):
        tl = WorkloadTimeline()
        # Build a 10s timeline so a 5s tail window discards the first half.
        # Each point is well-separated by 1s for deterministic anchor selection.
        tl.feed(_ok(start=0.0, completed=1.0, prompt=10, completion=10))   # t=1
        tl.feed(_ok(start=0.5, completed=3.0, prompt=10, completion=10))   # t=3
        tl.feed(_ok(start=1.0, completed=6.0, prompt=10, completion=10))   # t=6
        tl.feed(_ok(start=1.5, completed=10.0, prompt=10, completion=10))  # t=10
        # wall = 10s. Tail window = last 5s -> from t=5 anchor (latest with t<=5 is t=3).
        # delta_t = 10 - 3 = 7; delta_completion = 40 - 20 = 20
        rates = tl.last_window_rates(5.0)
        self.assertAlmostEqual(rates[3], 20 / 7, places=4)

    def test_steady_state_drops_warmup(self):
        tl = WorkloadTimeline()
        tl.feed(_ok(start=0.0, completed=1.0, prompt=10, completion=10))    # t=1
        tl.feed(_ok(start=0.5, completed=2.0, prompt=10, completion=10))    # t=2
        tl.feed(_ok(start=1.0, completed=8.0, prompt=10, completion=10))    # t=8
        tl.feed(_ok(start=1.5, completed=10.0, prompt=10, completion=10))   # t=10
        # wall = 10s, warmup_frac = 0.2 -> cutoff_t = 2s.
        # Anchor = latest point with t<=2 = the t=2 point. delta_t = 10-2 = 8;
        # delta_completion = 40 - 20 = 20.
        rates = tl.steady_state_rates(warmup_frac=0.2)
        self.assertAlmostEqual(rates[3], 20 / 8, places=4)

    def test_steady_state_falls_back_to_overall_when_anchor_is_last(self):
        # Single point can't carve a steady-state window.
        tl = WorkloadTimeline()
        tl.feed(_ok(start=0.0, completed=1.0, prompt=10, completion=10))
        self.assertEqual(tl.steady_state_rates(0.2), tl.overall_rates())


class TestThroughputSnapshot(unittest.TestCase):

    def test_to_summary_emits_four_rows(self):
        tl = WorkloadTimeline()
        tl.feed(_ok(start=0.0, completed=1.0, prompt=10, completion=10))
        tl.feed(_ok(start=0.5, completed=2.0, prompt=20, completion=20, cached=10))
        snap = tl.to_summary(last_window_s=30.0, warmup_frac=0.2)
        self.assertEqual(snap.n_samples, 2)
        self.assertEqual(len(snap.rows), 4)
        metrics = [r.metric for r in snap.rows]
        self.assertEqual(
            metrics,
            ['Total Prompt tok/s', 'New Prompt tok/s', 'Cached Prompt tok/s', 'Completion tok/s'],
        )

    def test_to_table_labels_reflect_window_and_warmup(self):
        tl = WorkloadTimeline()
        tl.feed(_ok(start=0.0, completed=1.0, prompt=10, completion=10))
        snap = tl.to_summary(last_window_s=60.0, warmup_frac=0.25)
        table = snap.to_table()
        self.assertIn('Overall', table)
        self.assertIn('Last 60s', table)
        self.assertIn('Steady (drop 25%)', table)

    def test_empty_summary_is_empty(self):
        snap = WorkloadThroughput()
        self.assertTrue(snap.is_empty())
        self.assertEqual(snap.to_table(), '')

    def test_raw_points_dict_shape(self):
        tl = WorkloadTimeline()
        tl.feed(_ok(start=0.0, completed=1.5, prompt=20, completion=10, cached=5))
        d = tl.to_raw_points_dict()
        self.assertIn('points', d)
        self.assertEqual(len(d['points']), 1)
        p = d['points'][0]
        for key in ('t', 'cum_completion', 'cum_new_prompt', 'cum_cached_prompt'):
            self.assertIn(key, p)
        self.assertAlmostEqual(p['t'], 1.5)
        self.assertEqual(p['cum_completion'], 10)


if __name__ == '__main__':
    unittest.main(buffer=False)
