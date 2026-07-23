# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for stream/non-stream metric bucketing.

Covers the is_stream decide-once classification, MetricsAccumulator bucketing,
and percentile path filtering — all as pure-logic tests with no model service
dependency (a mock SQLite result table is used for the percentile path).
"""
import os
import sqlite3
import tempfile
import unittest

from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator, is_stream_body
from evalscope.perf.utils.db_util import create_result_table, get_percentile_results, insert_benchmark_data


def _make(**kwargs):
    """Build a BenchmarkData with sensible streaming defaults."""
    data = BenchmarkData()
    data.success = kwargs.get('success', True)
    data.start_time = 0.0
    data.completed_time = 1.0
    data.query_latency = kwargs.get('latency', 1.0)
    data.first_chunk_latency = kwargs.get('fcl', 0.3)
    data.time_per_output_token = kwargs.get('tpot', 0.02)
    data.prompt_tokens = kwargs.get('prompt_tokens', 10)
    data.completion_tokens = kwargs.get('completion_tokens', 50)
    data.inter_chunk_latency = kwargs.get('itl', [0.02] * 49)
    data.decoded_tokens_per_iter = kwargs.get('decoded', 4.0)
    data.is_stream = kwargs.get('is_stream', True)
    data.request = kwargs.get('request', '{}')
    data.response_messages = []
    return data


class _DummyPlugin:
    def __init__(self, prompt, completion):
        self._p, self._c = prompt, completion

    def parse_responses(self, responses, request=None, **kw):
        return self._p, self._c


class TestIsStreamClassification(unittest.TestCase):
    """Tests for the is_stream_body() helper used on error/timeout paths."""

    def test_absent_is_nonstream(self):
        self.assertFalse(is_stream_body({}))

    def test_truthy_non_bool_is_nonstream(self):
        self.assertFalse(is_stream_body({'stream': 'true'}))
        self.assertFalse(is_stream_body({'stream': 'True'}))
        self.assertFalse(is_stream_body({'stream': 1}))


class TestAccumulatorBucketing(unittest.TestCase):

    def setUp(self):
        self.plugin = _DummyPlugin(10, 50)

    def test_stream_only_metrics_excludes_nonstream(self):
        acc = MetricsAccumulator()
        s1 = _make(fcl=0.3, latency=1.0, itl=[0.02] * 49, is_stream=True)
        s2 = _make(fcl=0.4, latency=1.0, itl=[0.04] * 49, is_stream=True)
        ns1 = _make(fcl=1.5, latency=1.5, itl=[], is_stream=False)
        ns2 = _make(fcl=1.6, latency=1.6, itl=[], is_stream=False)
        for d in (s1, s2, ns1, ns2):
            acc.update(d, self.plugin)
        result = acc.to_result()

        self.assertAlmostEqual(result.avg_first_chunk_latency, (0.3 + 0.4) / 2)
        # TPOT post-finalize: (1.0-0.3)/49, (1.0-0.4)/49
        exp_tpot = ((1.0 - 0.3) / 49 + (1.0 - 0.4) / 49) / 2
        self.assertAlmostEqual(result.avg_time_per_output_token, exp_tpot)
        self.assertAlmostEqual(result.avg_inter_token_latency, (0.02 * 49 + 0.04 * 49) / (2 * 49))

        # Generic latency averaged over all success (n=4)
        self.assertAlmostEqual(result.avg_latency, (1.0 + 1.0 + 1.5 + 1.6) / 4)

    def test_counts_include_failures(self):
        acc = MetricsAccumulator()
        acc.update(_make(is_stream=True), self.plugin)
        acc.update(_make(is_stream=True, success=False), self.plugin)
        acc.update(_make(is_stream=False), self.plugin)
        acc.update(_make(is_stream=False, success=False), self.plugin)
        result = acc.to_result()

        self.assertEqual(result.total_requests, 4)
        self.assertEqual(result.succeed_requests, 2)
        self.assertEqual(result.failed_requests, 2)
        self.assertEqual(result.stream_requests, 2)
        self.assertEqual(result.non_stream_requests, 2)

    def test_all_stream_no_regression(self):
        acc = MetricsAccumulator()
        acc.update(_make(fcl=0.3, latency=1.0, tpot=0.02, itl=[0.02] * 49, is_stream=True), self.plugin)
        acc.update(_make(fcl=0.5, latency=1.0, tpot=0.03, itl=[0.04] * 49, is_stream=True), self.plugin)
        result = acc.to_result()
        self.assertEqual(result.stream_requests, 2)
        self.assertEqual(result.non_stream_requests, 0)

    def test_all_stream_failed_empty_subset(self):
        acc = MetricsAccumulator()
        acc.update(_make(is_stream=True, success=False), self.plugin)
        acc.update(_make(is_stream=True, success=False), self.plugin)
        result = acc.to_result()
        self.assertEqual(result.stream_requests, 2)
        self.assertEqual(result.succeed_requests, 0)
        self.assertEqual(result.avg_first_chunk_latency, -1)
        self.assertEqual(result.avg_time_per_output_token, -1)

    def test_no_stream_falls_back_to_all_requests(self):
        # Pure non-stream run: streaming metrics fall back to the all-request
        # computation (backward compatible) rather than reporting -1.
        acc = MetricsAccumulator()
        acc.update(_make(fcl=1.5, latency=2.0, itl=[], is_stream=False), self.plugin)
        result = acc.to_result()
        self.assertEqual(result.avg_first_chunk_latency, 1.5)
        # finalize derives TPOT = (latency - fcl) / (completion_tokens - 1)
        self.assertAlmostEqual(result.avg_time_per_output_token, (2.0 - 1.5) / 49)
        self.assertEqual(result.avg_inter_token_latency, 0.0)


class TestPercentileBucketing(unittest.TestCase):

    def setUp(self):
        self.db = tempfile.mktemp(suffix='.db')
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        create_result_table(cur)
        # 4 stream (real TPOT) + 4 non-stream (TPOT 0, no ITL)
        for i in range(4):
            bd = _make(
                fcl=0.5, latency=2.0, tpot=0.03 * (i + 1), itl=[0.03] * 50,
                completion_tokens=51, prompt_tokens=10, is_stream=True,
            )
            insert_benchmark_data(cur, bd)
        for i in range(4):
            bd = _make(
                fcl=1.5, latency=1.5, tpot=0.0, itl=[], completion_tokens=51,
                prompt_tokens=10, is_stream=False,
            )
            insert_benchmark_data(cur, bd)
        con.commit()
        con.close()

    def tearDown(self):
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_stream_only_columns_no_zero_tpot(self):
        result = get_percentile_results(self.db, api_type='openai')
        rows = result.to_list()
        for p in ['1%', '50%', '75%', '99%']:
            row = next(r for r in rows if r['Percentiles'] == p)
            tpot = row['TPOT (ms)']
            self.assertNotEqual(tpot, 0.0, f'TPOT at {p} must not be 0 (non-stream excluded)')
            self.assertEqual(row['ITL (ms)'], 30.0, f'ITL at {p} must be 30ms (stream-only)')

    def test_pure_non_stream_percentiles_fall_back_to_all_rows(self):
        # Pure non-stream run: no stream rows, so streaming metrics fall back to
        # all rows (backward compatible) instead of producing NaN.
        db = tempfile.mktemp(suffix='.db')
        con = sqlite3.connect(db)
        cur = con.cursor()
        create_result_table(cur)
        for _ in range(3):
            bd = _make(
                fcl=1.5, latency=1.5, tpot=0.0, itl=[], completion_tokens=51,
                prompt_tokens=10, is_stream=False,
            )
            insert_benchmark_data(cur, bd)
        con.commit()
        con.close()
        try:
            rows = get_percentile_results(db, api_type='openai').to_list()
            p50 = next(r for r in rows if r['Percentiles'] == '50%')
            self.assertEqual(p50['TTFT (ms)'], 1500.0)
            self.assertEqual(p50['TPOT (ms)'], 0.0)
        finally:
            if os.path.exists(db):
                os.unlink(db)


class TestSingleTurnCachedTokenSync(unittest.TestCase):
    """Cache-hit normalization for single-turn (open-loop / closed-loop) runs.

    Regression coverage for issue #1506: server-reported cached tokens land in
    ``real_cached_tokens`` but were never synced to ``cached_tokens``, so single-turn
    cache metrics stayed at 0.
    """

    def setUp(self):
        self.plugin = _DummyPlugin(152, 24)

    def test_finalize_syncs_real_cached_tokens(self):
        data = _make(prompt_tokens=152, completion_tokens=24, is_stream=False)
        data.real_cached_tokens = 128
        data.finalize(self.plugin)
        self.assertEqual(data.cached_tokens, 128)

    def test_finalize_does_not_overwrite_existing_cached_tokens(self):
        data = _make(prompt_tokens=152, completion_tokens=24, is_stream=False)
        data.real_cached_tokens = 128
        data.cached_tokens = 0
        data.finalize(self.plugin)
        self.assertEqual(data.cached_tokens, 0)

    def test_accumulator_reports_cached_percent_for_single_turn(self):
        acc = MetricsAccumulator()
        data = _make(prompt_tokens=152, completion_tokens=24, is_stream=False)
        data.real_cached_tokens = 128
        acc.update(data, self.plugin)
        result = acc.to_result()
        self.assertAlmostEqual(result.avg_cached_percent, 128 / 152 * 100.0)


if __name__ == '__main__':
    unittest.main()
