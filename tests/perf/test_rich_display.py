# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the refactored LLM rich display tables."""
import io
import unittest
from rich.console import Console

from evalscope.perf.utils.perf_models import BenchmarkSummary, PercentileResult, PercentileRow
from evalscope.perf.utils.rich_display import AnalysisResult, DualConsole, LLMSummaryRenderer
from evalscope.perf.utils.trace_metrics import TraceLevelSummary, TraceMetricStats
from evalscope.perf.utils.workload_timeline import WorkloadThroughput, WorkloadThroughputRow


def _make_summary(**overrides) -> BenchmarkSummary:
    defaults = dict(
        time_taken=10.0,
        concurrency=2,
        request_rate=-1,
        total_requests=20,
        succeed_requests=20,
        failed_requests=0,
        request_throughput=2.0,
        avg_latency=1.5,
        avg_input_tokens=500.0,
        output_token_throughput=50.0,
        total_token_throughput=550.0,
        avg_ttft=100.0,
        avg_tpot=20.0,
        avg_itl=18.0,
        avg_output_tokens=100.0,
        input_token_throughput=500.0,
    )
    defaults.update(overrides)
    return BenchmarkSummary(**defaults)


def _make_percentiles() -> PercentileResult:
    return PercentileResult(rows=[
        PercentileRow(percentile='50%', latency=1.2, ttft=80.0, tpot=18.0,
                      input_tokens=450.0, output_tokens=90.0),
        PercentileRow(percentile='99%', latency=3.5, ttft=200.0, tpot=35.0,
                      input_tokens=900.0, output_tokens=300.0),
        PercentileRow(percentile='max', latency=5.0, ttft=350.0, tpot=50.0,
                      input_tokens=1024.0, output_tokens=512.0),
    ])


def _make_trace_summary() -> TraceLevelSummary:
    return TraceLevelSummary(n_traces=3, rows=[
        TraceMetricStats(metric='Latency (s)', mean=51.3, min=13.8, p50=53.1,
                         p90=80.2, p95=83.6, p99=86.3, max=87.0),
        TraceMetricStats(metric='Cache Hit Rate (%)', mean=36.0, min=24.3, p50=41.4,
                         p90=42.1, p95=42.1, p99=42.2, max=42.2),
    ])


def _make_workload() -> WorkloadThroughput:
    return WorkloadThroughput(n_samples=29, wall_time_s=87.0, last_window_s=30.0, warmup_frac=0.2, rows=[
        WorkloadThroughputRow(metric='Total Prompt tok/s', overall=3118.7, last_window=3028.2, steady_state=3035.0),
        WorkloadThroughputRow(metric='Completion tok/s', overall=46.2, last_window=48.2, steady_state=45.2),
    ])


def _build_all_results(configs):
    """Build all_results dict from a list of config dicts.

    Each config dict may contain: summary, percentiles, trace_summary, workload_throughput.
    Keys are auto-generated as parallel_N_number_M.
    """
    results = {}
    for i, cfg in enumerate(configs):
        summary = cfg.get('summary', _make_summary(concurrency=2 * (i + 1)))
        key = f'parallel_{summary.concurrency}_number_{summary.total_requests}'
        entry = {
            'metrics': summary,
            'percentiles': cfg.get('percentiles', _make_percentiles()),
        }
        if 'trace_summary' in cfg:
            entry['trace_summary'] = cfg['trace_summary']
        if 'workload_throughput' in cfg:
            entry['workload_throughput'] = cfg['workload_throughput']
        results[key] = entry
    return results


def _capture(all_results) -> str:
    """Render all_results through LLMSummaryRenderer and return captured text."""
    renderer = LLMSummaryRenderer()
    entries = renderer._collect_entries(all_results)

    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    devnull = Console(file=io.StringIO(), width=120, force_terminal=False)
    dc = DualConsole(console, devnull)

    renderer._render_overview_table(entries, dc)
    renderer._render_per_request_metrics(entries, dc)
    renderer._render_per_trace_metrics(entries, dc)
    renderer._render_workload_throughput(entries, dc)
    return buf.getvalue()


class TestOverviewTable(unittest.TestCase):

    def test_single_run(self):
        results = _build_all_results([{'summary': _make_summary()}])
        text = _capture(results)
        self.assertIn('Performance Overview', text)
        self.assertIn('2', text)
        self.assertIn('2.00', text)
        self.assertIn('50.00', text)
        self.assertIn('100.0%', text)

    def test_sweep(self):
        results = _build_all_results([
            {'summary': _make_summary(concurrency=2)},
            {'summary': _make_summary(concurrency=4, total_requests=40, succeed_requests=40)},
        ])
        text = _capture(results)
        self.assertIn('2', text)
        self.assertIn('4', text)

    def test_open_loop(self):
        results = _build_all_results([{
            'summary': _make_summary(concurrency=-1, request_rate=5.0),
        }])
        text = _capture(results)
        self.assertIn('5.00', text)

    def test_traces_column(self):
        results = _build_all_results([{
            'summary': _make_summary(),
            'trace_summary': _make_trace_summary(),
        }])
        text = _capture(results)
        self.assertIn('Traces', text)
        self.assertIn('3', text)


class TestPerRequestMetrics(unittest.TestCase):

    def test_single_run(self):
        results = _build_all_results([{'summary': _make_summary()}])
        text = _capture(results)
        self.assertIn('Per-Request Metrics', text)
        self.assertIn('Latency (s)', text)
        self.assertIn('TTFT (ms)', text)
        self.assertIn('TPOT (ms)', text)
        self.assertIn('Input Tokens', text)
        self.assertIn('Output Tokens', text)
        # max column header and max latency value from fixture
        lines = [l for l in text.splitlines() if 'max' in l.lower()]
        self.assertTrue(any('max' in l for l in lines), 'max column should appear in Per-Request Metrics')
        self.assertIn('5.000', text)  # max latency from _make_percentiles

    def test_multi_turn_rows(self):
        results = _build_all_results([{
            'summary': _make_summary(
                avg_turns=3.5,
                avg_cached_percent=42.0,
                avg_first_turn_ttft=500.0,
                avg_subsequent_turn_ttft=100.0,
            ),
        }])
        text = _capture(results)
        self.assertIn('Turns/Req', text)
        self.assertIn('Cache Hit (%)', text)
        self.assertIn('1st-Turn TTFT (ms)', text)
        self.assertIn('Subseq. TTFT (ms)', text)

    def test_spec_decode_rows(self):
        results = _build_all_results([{
            'summary': _make_summary(
                avg_decoded_tokens_per_iter=3.5,
                approx_spec_acceptance_rate=0.7,
            ),
        }])
        text = _capture(results)
        self.assertIn('Decoded Tok/Iter', text)
        self.assertIn('Spec. Accept Rate', text)
        self.assertIn('70.0%', text)

    def test_decode_tps_shown(self):
        results = _build_all_results([{'summary': _make_summary(avg_tpot=20.0)}])
        text = _capture(results)
        self.assertIn('Decode toks/s', text)
        self.assertIn('50.00', text)

    def test_no_ttft_when_zero(self):
        results = _build_all_results([{
            'summary': _make_summary(avg_ttft=0.0, avg_tpot=0.0),
            'percentiles': PercentileResult(rows=[
                PercentileRow(percentile='50%', latency=1.2, input_tokens=450.0),
                PercentileRow(percentile='99%', latency=3.5, input_tokens=900.0),
            ]),
        }])
        text = _capture(results)
        self.assertNotIn('TTFT (ms)', text)
        self.assertNotIn('TPOT (ms)', text)


class TestPerTraceMetrics(unittest.TestCase):

    def test_single_turn_not_shown(self):
        results = _build_all_results([{'summary': _make_summary()}])
        text = _capture(results)
        self.assertNotIn('Per-Trace Metrics', text)

    def test_multi_turn_shown(self):
        results = _build_all_results([{
            'summary': _make_summary(),
            'trace_summary': _make_trace_summary(),
        }])
        text = _capture(results)
        self.assertIn('Per-Trace Metrics', text)
        self.assertIn('Latency (s)', text)
        self.assertIn('Cache Hit Rate (%)', text)
        self.assertIn('51.30', text)

    def test_sweep_merged(self):
        results = _build_all_results([
            {'summary': _make_summary(concurrency=2), 'trace_summary': _make_trace_summary()},
            {'summary': _make_summary(concurrency=4, total_requests=40, succeed_requests=40),
             'trace_summary': _make_trace_summary()},
        ])
        text = _capture(results)
        count = text.count('Per-Trace Metrics')
        self.assertEqual(count, 1)


class TestWorkloadThroughput(unittest.TestCase):

    def test_single_run(self):
        results = _build_all_results([{
            'summary': _make_summary(),
            'workload_throughput': _make_workload(),
        }])
        text = _capture(results)
        self.assertIn('Workload Throughput', text)
        self.assertIn('Total Prompt tok/s', text)
        self.assertIn('Completion tok/s', text)
        self.assertIn('3118.70', text)
        self.assertIn('Last 30s', text)

    def test_sweep_merged(self):
        results = _build_all_results([
            {'summary': _make_summary(concurrency=2), 'workload_throughput': _make_workload()},
            {'summary': _make_summary(concurrency=4, total_requests=40, succeed_requests=40),
             'workload_throughput': _make_workload()},
        ])
        text = _capture(results)
        count = text.count('Workload Throughput')
        self.assertEqual(count, 1)

    def test_not_shown_when_absent(self):
        results = _build_all_results([{'summary': _make_summary()}])
        text = _capture(results)
        self.assertNotIn('Workload Throughput', text)


class TestCollectEntries(unittest.TestCase):

    def test_sorted_by_concurrency(self):
        results = _build_all_results([
            {'summary': _make_summary(concurrency=4, total_requests=40, succeed_requests=40)},
            {'summary': _make_summary(concurrency=2)},
        ])
        entries = LLMSummaryRenderer._collect_entries(results)
        concurrencies = [e[0].concurrency for e in entries]
        self.assertEqual(concurrencies, [2, 4])

    def test_open_loop_sorted_by_rate(self):
        results = _build_all_results([
            {'summary': _make_summary(concurrency=-1, request_rate=10.0)},
            {'summary': _make_summary(concurrency=-1, request_rate=5.0, total_requests=10, succeed_requests=10)},
        ])
        entries = LLMSummaryRenderer._collect_entries(results)
        rates = [e[0].request_rate for e in entries]
        self.assertEqual(rates, [5.0, 10.0])


if __name__ == '__main__':
    unittest.main()
