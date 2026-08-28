"""Unit tests for failure-record timing in perf metrics (wall_time / QPS).

``AioHttpClient.post`` used to build untimed failure records
(``start_time=0.0``), and ``MetricsAccumulator._update_wall_time`` folded them
into the wall-clock window via ``min(...)``.  A single failed request therefore
pinned ``_wall_start`` to 0.0 (the perf_counter epoch), inflating wall_time to
hours and collapsing QPS to ~0.  Failure records must carry a real timestamp,
and untimed records must never widen the window.
"""
import asyncio
import time
from typing import Any

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator


class _DummyPlugin:
    """Minimal api plugin: finalize() only needs parse_responses()."""

    def parse_responses(self, responses, request=None, **kwargs):
        return 10, 5


def _success(start: float, end: float) -> BenchmarkData:
    data = BenchmarkData(
        success=True,
        start_time=start,
        completed_time=end,
        query_latency=end - start,
        first_chunk_latency=0.2,
        prompt_tokens=10,
        completion_tokens=5,
        is_stream=True,
    )
    data.request = '{}'
    data.response_messages = []
    return data


class TestWallTimeWithFailedRequests:

    def test_untimed_failure_does_not_corrupt_wall_time(self):
        accumulator = MetricsAccumulator()
        accumulator.update(_success(10.0, 11.0), _DummyPlugin())
        # Untimed failure record, as produced by legacy code paths.
        accumulator.update(BenchmarkData(success=False, start_time=0.0, completed_time=0.0), _DummyPlugin())

        result = accumulator.to_result()

        assert accumulator.wall_time == pytest.approx(1.0)
        assert result.qps == pytest.approx(1.0)
        assert result.failed_requests == 1

    def test_timestamped_failure_still_expands_window(self):
        accumulator = MetricsAccumulator()
        accumulator.update(_success(10.0, 11.0), _DummyPlugin())
        # A properly stamped failure widens the window like any real request.
        accumulator.update(BenchmarkData(success=False, start_time=9.5, completed_time=11.5), _DummyPlugin())

        assert accumulator.wall_time == pytest.approx(2.0)

    def test_only_untimed_failures_keeps_guard_wall_time(self):
        # Before any timed record arrives the guard value must stay in place.
        accumulator = MetricsAccumulator()
        accumulator.update(BenchmarkData(success=False, start_time=0.0, completed_time=0.0), _DummyPlugin())

        assert accumulator.wall_time == 1.0

    def test_incomplete_failure_does_not_corrupt_wall_time(self) -> None:
        accumulator = MetricsAccumulator()
        accumulator.update(_success(10.0, 11.0), _DummyPlugin())
        accumulator.update(BenchmarkData(success=False, start_time=12.0, completed_time=0.0), _DummyPlugin())

        assert accumulator.wall_time == pytest.approx(1.0)


class _ExplodingPlugin:
    """Api plugin whose process_request() raises, e.g. a buggy custom plugin."""

    def extract_body_meta(self, body, headers):
        return headers, None

    async def process_request(self, client_session, url, headers, body):
        raise RuntimeError('plugin exploded')

    def parse_responses(self, responses: Any, request: Any = None, **kwargs: Any) -> tuple[int, int]:
        return 10, 5


class _SlowFailureThenSuccessPlugin(_ExplodingPlugin):

    def __init__(self) -> None:
        self.calls = 0

    async def process_request(self, client_session: Any, url: str, headers: dict, body: Any) -> BenchmarkData:
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(0.06)
            raise RuntimeError('slow failure')

        start = time.perf_counter()
        await asyncio.sleep(0.01)
        return _success(start, time.perf_counter())


class _ReturningSlowFailurePlugin(_ExplodingPlugin):

    async def process_request(self, client_session: Any, url: str, headers: dict, body: Any) -> BenchmarkData:
        start = time.perf_counter()
        await asyncio.sleep(0.06)
        return BenchmarkData(success=False, error='HTTP 500', start_time=start)


class TestHttpClientFailureRecordsAreTimestamped:

    def test_slow_failure_is_included_in_wall_time_and_qps(self) -> None:
        async def run() -> tuple[BenchmarkData, BenchmarkData]:
            args = Arguments(model='test-model', api='openai')
            args.parallel = 1
            client = AioHttpClient(args, _SlowFailureThenSuccessPlugin())
            async with client:
                failure = await client.post({'stream': True})
                success = await client.post({'stream': True})
            return failure, success

        failure, success = asyncio.run(run())
        accumulator = MetricsAccumulator()
        accumulator.update(failure, _DummyPlugin())
        accumulator.update(success, _DummyPlugin())
        result = accumulator.to_result()
        expected_wall_time = success.completed_time - failure.start_time

        assert failure.completed_time - failure.start_time >= 0.05
        assert result.total_time == pytest.approx(expected_wall_time)
        assert result.qps == pytest.approx(1 / expected_wall_time)

    def test_plugin_failure_gets_a_completion_timestamp(self) -> None:
        async def run() -> BenchmarkData:
            args = Arguments(model='test-model', api='openai')
            args.parallel = 1
            client = AioHttpClient(args, _ReturningSlowFailurePlugin())
            async with client:
                return await client.post({'stream': True})

        failure = asyncio.run(run())

        assert failure.completed_time - failure.start_time >= 0.05

    @pytest.mark.parametrize('error', [RuntimeError('plugin exploded'), asyncio.TimeoutError()])
    def test_failure_record_carries_perf_counter_timestamp(self, error):
        async def run() -> BenchmarkData:
            args = Arguments(model='test-model', api='openai')
            # run_one_benchmark() collapses sweep lists to scalars before the
            # HTTP client is constructed; mirror that here.
            args.parallel = 1
            client = AioHttpClient(args, _ExplodingPlugin())
            client.api_plugin.process_request = self._raising(error)
            async with client:
                return await client.post({'stream': True})

        before = time.perf_counter()
        data = asyncio.run(run())
        after = time.perf_counter()

        assert data.success is False
        assert data.is_stream is True
        assert before <= data.start_time <= after
        assert data.completed_time >= data.start_time

    @staticmethod
    def _raising(error: BaseException):
        async def process_request(client_session, url, headers, body):
            raise error

        return process_request
