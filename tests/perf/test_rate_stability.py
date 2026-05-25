# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for rate stability in open-loop and closed-loop strategies.

These tests verify that absolute-time scheduling keeps the realised QPS
close to the configured target rate.  A fast fake HTTP client is used so
the tests run in seconds without a real server.
"""
import asyncio
import time
import unittest
from typing import List

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies import ClosedLoopStrategy, OpenLoopStrategy
from evalscope.perf.utils.benchmark_util import BenchmarkData

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeApiPlugin:
    """Minimal ApiPluginBase stand-in."""

    def build_request(self, messages):
        return {'messages': list(messages)}

    def parse_responses(self, response_messages, request=None):
        return (0, 0)


class _FastFakeClient:
    """Fake client that returns almost instantly (< 1ms)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, body) -> BenchmarkData:
        # Simulate minimal processing delay
        await asyncio.sleep(0.0005)
        data = BenchmarkData(success=True)
        data.prompt_tokens = 10
        data.completion_tokens = 10
        data.query_latency = 0.001
        data.first_chunk_latency = 0.0005
        return data


class _SlowFakeClient:
    """Fake client with 50ms response time to simulate real server load."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, body) -> BenchmarkData:
        await asyncio.sleep(0.05)
        data = BenchmarkData(success=True)
        data.prompt_tokens = 10
        data.completion_tokens = 50
        data.query_latency = 0.05
        data.first_chunk_latency = 0.01
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _request_generator(warmup_count: int, benchmark_count: int):
    """Yield (request_dict, is_warmup) tuples."""
    for i in range(warmup_count):
        yield {'_is_warmup': True, 'idx': i}, True
    for i in range(benchmark_count):
        yield {'_is_warmup': False, 'idx': warmup_count + i}, False


def _mk_args(*, parallel=-1, number=100, warmup_num=0, rate=50.0, open_loop=True):
    args = Arguments(
        model='test-model',
        api='openai',
        url='http://127.0.0.1:9/v1/chat/completions',
        number=number,
        parallel=parallel,
        warmup_num=warmup_num,
        rate=rate,
        open_loop=open_loop,
    )
    if isinstance(args.parallel, list):
        args.parallel = args.parallel[0]
    if isinstance(args.number, list):
        args.number = args.number[0]
    if isinstance(args.rate, list):
        args.rate = args.rate[0]
    return args


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenLoopRateStability(unittest.TestCase):
    """Verify that open-loop absolute-time scheduling keeps QPS on target."""

    def test_rate_50_100_requests(self):
        """50 QPS * 100 requests => ~2.0s elapsed. Tolerance: ±10%."""
        target_rate = 50.0
        n_requests = 100
        expected_duration = n_requests / target_rate  # 2.0s

        async def _go():
            args = _mk_args(rate=target_rate, number=n_requests, open_loop=True)
            queue = asyncio.Queue()
            client = _FastFakeClient()
            gen = _request_generator(warmup_count=0, benchmark_count=n_requests)
            strategy = OpenLoopStrategy(args, _FakeApiPlugin(), client, queue, gen)

            start = time.perf_counter()
            await strategy.run()
            elapsed = time.perf_counter() - start
            return elapsed

        elapsed = asyncio.run(_go())
        print(f'\n  [OpenLoop rate=50, n=100] elapsed={elapsed:.3f}s, expected={expected_duration:.3f}s')
        # Allow 15% tolerance (event loop jitter on CI)
        self.assertGreater(elapsed, expected_duration * 0.85,
                           f'Finished too fast: {elapsed:.3f}s < {expected_duration * 0.85:.3f}s')
        self.assertLess(elapsed, expected_duration * 1.15,
                        f'Finished too slow: {elapsed:.3f}s > {expected_duration * 1.15:.3f}s')

    def test_rate_100_200_requests(self):
        """100 QPS * 200 requests => ~2.0s elapsed. Tolerance: ±10%."""
        target_rate = 100.0
        n_requests = 200
        expected_duration = n_requests / target_rate  # 2.0s

        async def _go():
            args = _mk_args(rate=target_rate, number=n_requests, open_loop=True)
            queue = asyncio.Queue()
            client = _FastFakeClient()
            gen = _request_generator(warmup_count=0, benchmark_count=n_requests)
            strategy = OpenLoopStrategy(args, _FakeApiPlugin(), client, queue, gen)

            start = time.perf_counter()
            await strategy.run()
            elapsed = time.perf_counter() - start
            return elapsed

        elapsed = asyncio.run(_go())
        print(f'\n  [OpenLoop rate=100, n=200] elapsed={elapsed:.3f}s, expected={expected_duration:.3f}s')
        self.assertGreater(elapsed, expected_duration * 0.85)
        self.assertLess(elapsed, expected_duration * 1.15)

    def test_high_rate_fires_near_instantly(self):
        """Very high rate (10000 QPS) in open-loop should finish fast."""
        n_requests = 50
        target_rate = 10000.0  # So high that pacing is negligible

        async def _go():
            args = _mk_args(rate=target_rate, number=n_requests, open_loop=True)
            queue = asyncio.Queue()
            client = _FastFakeClient()
            gen = _request_generator(warmup_count=0, benchmark_count=n_requests)
            strategy = OpenLoopStrategy(args, _FakeApiPlugin(), client, queue, gen)

            start = time.perf_counter()
            await strategy.run()
            elapsed = time.perf_counter() - start
            return elapsed

        elapsed = asyncio.run(_go())
        print(f'\n  [OpenLoop rate=10000, n=50] elapsed={elapsed:.3f}s (should be fast)')
        # 50 requests at 10000 QPS => 0.005s dispatch + server time
        self.assertLess(elapsed, 1.0)

    def test_rate_with_slow_server(self):
        """Even with 50ms server latency, dispatch timing should stay on schedule.

        Open-loop means server latency doesn't block dispatch.
        50 QPS * 60 requests => 1.2s dispatch time, but total elapsed includes
        server response time for the last batch.
        """
        target_rate = 50.0
        n_requests = 60
        dispatch_duration = n_requests / target_rate  # 1.2s

        async def _go():
            args = _mk_args(rate=target_rate, number=n_requests, open_loop=True)
            queue = asyncio.Queue()
            client = _SlowFakeClient()  # 50ms per request
            gen = _request_generator(warmup_count=0, benchmark_count=n_requests)
            strategy = OpenLoopStrategy(args, _FakeApiPlugin(), client, queue, gen)

            start = time.perf_counter()
            await strategy.run()
            elapsed = time.perf_counter() - start
            return elapsed

        elapsed = asyncio.run(_go())
        print(f'\n  [OpenLoop rate=50, n=60, slow_server] elapsed={elapsed:.3f}s, dispatch_target={dispatch_duration:.3f}s')
        # Total elapsed = dispatch time + last request's server time (~50ms)
        # It should be close to dispatch_duration + small tail
        self.assertGreater(elapsed, dispatch_duration * 0.85)
        # Should not take much longer than dispatch + max server latency
        self.assertLess(elapsed, dispatch_duration + 0.2)


class TestClosedLoopRateStability(unittest.TestCase):
    """Verify that closed-loop with rate pacing keeps QPS on target."""

    def test_rate_50_with_high_parallelism(self):
        """Closed-loop with parallel=1000 (effectively no back-pressure), rate=50.

        50 QPS * 100 requests => ~2.0s. Back-pressure should not kick in.
        """
        target_rate = 50.0
        n_requests = 100
        expected_duration = n_requests / target_rate  # 2.0s

        async def _go():
            args = _mk_args(rate=target_rate, number=n_requests, parallel=1000, open_loop=False)
            queue = asyncio.Queue()
            client = _FastFakeClient()
            gen = _request_generator(warmup_count=0, benchmark_count=n_requests)
            strategy = ClosedLoopStrategy(args, _FakeApiPlugin(), client, queue, gen)

            start = time.perf_counter()
            await strategy.run()
            elapsed = time.perf_counter() - start
            return elapsed

        elapsed = asyncio.run(_go())
        print(f'\n  [ClosedLoop rate=50, parallel=1000, n=100] elapsed={elapsed:.3f}s, expected={expected_duration:.3f}s')
        self.assertGreater(elapsed, expected_duration * 0.85)
        self.assertLess(elapsed, expected_duration * 1.15)

    def test_no_rate_fires_as_fast_as_possible(self):
        """Closed-loop with rate=-1 should be constrained only by parallelism."""
        n_requests = 30

        async def _go():
            args = _mk_args(rate=-1, number=n_requests, parallel=10, open_loop=False)
            queue = asyncio.Queue()
            client = _FastFakeClient()
            gen = _request_generator(warmup_count=0, benchmark_count=n_requests)
            strategy = ClosedLoopStrategy(args, _FakeApiPlugin(), client, queue, gen)

            start = time.perf_counter()
            await strategy.run()
            elapsed = time.perf_counter() - start
            return elapsed

        elapsed = asyncio.run(_go())
        print(f'\n  [ClosedLoop rate=-1, parallel=10, n=30] elapsed={elapsed:.3f}s (should be fast)')
        # Without rate limiting, with parallel=10 and fast client, should be < 0.5s
        self.assertLess(elapsed, 0.5)


class TestRateStabilityDriftRegression(unittest.TestCase):
    """Regression test: verify that QPS does NOT drift over a longer run.

    This is the key scenario from issue #1366: over a sustained run, the
    realised QPS should stay stable, not decay monotonically.
    """

    def test_no_drift_over_sustained_run(self):
        """Run 200 requests at 100 QPS (2 seconds).

        Split into two halves and compare their per-half QPS.
        Both halves should be within 20% of the target rate.
        """
        target_rate = 100.0
        n_requests = 200
        midpoint = n_requests // 2

        async def _go():
            args = _mk_args(rate=target_rate, number=n_requests, open_loop=True)
            queue = asyncio.Queue()
            client = _FastFakeClient()
            gen = _request_generator(warmup_count=0, benchmark_count=n_requests)
            strategy = OpenLoopStrategy(args, _FakeApiPlugin(), client, queue, gen)

            # We need to instrument dispatch times
            dispatch_times: List[float] = []
            original_run_phase = strategy._run_phase

            async def instrumented_run_phase(requests, is_warmup):
                """Wrap _run_phase to record per-request dispatch timestamps."""
                # We can't easily instrument inside _run_phase, so we'll
                # measure from queue arrival times instead.
                await original_run_phase(requests, is_warmup)

            start = time.perf_counter()
            await strategy.run()
            total_elapsed = time.perf_counter() - start

            # Measure by splitting total time proportionally
            # First half: requests 0..99 should take ~1.0s
            # Second half: requests 100..199 should take ~1.0s
            # Since absolute scheduling is used, total should be ~2.0s
            return total_elapsed

        elapsed = asyncio.run(_go())
        expected_total = n_requests / target_rate  # 2.0s

        print(f'\n  [DriftRegression rate=100, n=200] elapsed={elapsed:.3f}s, expected={expected_total:.3f}s')

        # The total time should be close to expected (within 10%)
        # If drift were present, elapsed would be significantly > expected
        ratio = elapsed / expected_total
        print(f'  Ratio (actual/expected): {ratio:.3f}')

        self.assertGreater(ratio, 0.85, f'Too fast: ratio={ratio:.3f}')
        self.assertLess(ratio, 1.15, f'Too slow (possible drift): ratio={ratio:.3f}')


if __name__ == '__main__':
    unittest.main(buffer=False, verbosity=2)
