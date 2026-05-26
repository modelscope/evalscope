# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for first-turn vs subsequent-turn TTFT bucketing (P2).

Covers ``MetricsAccumulator``'s split of TTFT by turn position and the
``BenchmarkSummary`` rendering of the resulting averages.  Single-turn
requests (``input_num_turns == 0``) must leave both buckets empty so the
cold/warm distinction stays meaningful for non-multi-turn benchmarks.
"""
import unittest

from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator
from evalscope.perf.utils.perf_constants import Metrics
from evalscope.perf.utils.perf_models import BenchmarkSummary


class _NoopApi:
    """``MetricsAccumulator.update`` calls ``data.finalize(api_plugin)``; we
    pre-populate every field that finalize would compute so it becomes a no-op."""

    def parse_responses(self, response_messages, request=None):  # pragma: no cover
        return (0, 0)


def _make_data(*, ttft: float, is_first: bool, input_num_turns: int, prompt_tokens: int = 100) -> BenchmarkData:
    d = BenchmarkData(success=True)
    d.start_time = 0.0
    d.completed_time = 1.0
    d.query_latency = 1.0
    d.first_chunk_latency = ttft
    d.prompt_tokens = prompt_tokens
    d.completion_tokens = 50
    d.is_first_turn = is_first
    d.input_num_turns = input_num_turns
    return d


class TestFirstTurnSplit(unittest.TestCase):

    def test_multi_turn_buckets_split_correctly(self):
        acc = MetricsAccumulator()
        # 2 first turns @ 0.5s, 4 subsequent turns @ 0.05s
        acc.update(_make_data(ttft=0.5, is_first=True, input_num_turns=1), _NoopApi())
        acc.update(_make_data(ttft=0.5, is_first=True, input_num_turns=1), _NoopApi())
        for _ in range(4):
            acc.update(_make_data(ttft=0.05, is_first=False, input_num_turns=2), _NoopApi())

        self.assertEqual(acc.n_first_turn, 2)
        self.assertEqual(acc.n_subsequent_turn, 4)
        self.assertAlmostEqual(acc.total_first_turn_ttft, 1.0)
        self.assertAlmostEqual(acc.total_subsequent_turn_ttft, 0.2)

        result = acc.to_result()
        self.assertAlmostEqual(result.avg_first_turn_ttft, 0.5)
        self.assertAlmostEqual(result.avg_subsequent_turn_ttft, 0.05)

    def test_single_turn_leaves_buckets_empty(self):
        acc = MetricsAccumulator()
        # input_num_turns == 0 marks a single-turn (non-conversation) request.
        for _ in range(3):
            acc.update(_make_data(ttft=0.1, is_first=False, input_num_turns=0), _NoopApi())

        self.assertEqual(acc.n_first_turn, 0)
        self.assertEqual(acc.n_subsequent_turn, 0)

        result = acc.to_result()
        # Sentinel -1 means "not applicable"; downstream rendering must skip these.
        self.assertEqual(result.avg_first_turn_ttft, -1)
        self.assertEqual(result.avg_subsequent_turn_ttft, -1)

    def test_message_includes_ms_fields_when_multi_turn(self):
        acc = MetricsAccumulator()
        acc.update(_make_data(ttft=0.4, is_first=True, input_num_turns=1), _NoopApi())
        acc.update(_make_data(ttft=0.04, is_first=False, input_num_turns=2), _NoopApi())

        msg = acc.to_result().create_message(api_type='openai')
        self.assertIn(Metrics.AVERAGE_FIRST_TURN_TTFT, msg)
        self.assertIn(Metrics.AVERAGE_SUBSEQUENT_TURN_TTFT, msg)
        # Seconds -> milliseconds at the message layer.
        self.assertAlmostEqual(msg[Metrics.AVERAGE_FIRST_TURN_TTFT], 400.0, places=2)
        self.assertAlmostEqual(msg[Metrics.AVERAGE_SUBSEQUENT_TURN_TTFT], 40.0, places=2)

    def test_message_omits_ms_fields_for_single_turn(self):
        acc = MetricsAccumulator()
        for _ in range(3):
            acc.update(_make_data(ttft=0.1, is_first=False, input_num_turns=0), _NoopApi())

        msg = acc.to_result().create_message(api_type='openai')
        self.assertNotIn(Metrics.AVERAGE_FIRST_TURN_TTFT, msg)
        self.assertNotIn(Metrics.AVERAGE_SUBSEQUENT_TURN_TTFT, msg)

    def test_summary_round_trip_and_table_render(self):
        acc = MetricsAccumulator()
        acc.update(_make_data(ttft=0.4, is_first=True, input_num_turns=1), _NoopApi())
        acc.update(_make_data(ttft=0.04, is_first=False, input_num_turns=2), _NoopApi())
        msg = acc.to_result().create_message(api_type='openai')

        summary = BenchmarkSummary.from_dict(msg)
        self.assertAlmostEqual(summary.avg_first_turn_ttft, 400.0, places=2)
        self.assertAlmostEqual(summary.avg_subsequent_turn_ttft, 40.0, places=2)

        table = summary.to_table()
        self.assertIn('Multi-turn', table)
        self.assertIn(Metrics.AVERAGE_FIRST_TURN_TTFT, table)
        self.assertIn(Metrics.AVERAGE_SUBSEQUENT_TURN_TTFT, table)


if __name__ == '__main__':
    unittest.main(buffer=False)
