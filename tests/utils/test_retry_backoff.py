"""Tests for transport-retry backoff (issue: honor Retry-After, bounded jittered backoff).

Delays are verified with an injected sleep — no global monkeypatching and no real waits.
"""
import httpx
import pytest
from openai import AuthenticationError, RateLimitError

from evalscope.utils.function_utils import async_retry_call, retry_call


class Recorder:
    """Fake sleep that records delays and a callable that fails N times then succeeds."""

    def __init__(self, fail_times: int, exc_factory=None):
        self.fail_times = fail_times
        self.calls = 0
        self.delays = []
        self.exc_factory = exc_factory or (lambda: RateLimitError('limited', response=_resp(429), body=None))

    def func(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc_factory()
        return 'ok'

    def sleep_sync(self, delay):
        self.delays.append(delay)

    async def sleep_async(self, delay):
        self.delays.append(delay)


def _resp(status, headers=None):
    return httpx.Response(status_code=status, headers=headers or {}, request=httpx.Request('POST', 'http://x/v1'))


def test_fixed_backoff_is_unchanged():
    rec = Recorder(fail_times=2)
    result = retry_call(rec.func, retries=4, sleep_interval=10, sleep=rec.sleep_sync)

    assert result == 'ok'
    assert rec.delays == [10, 10]  # flat cadence, byte-for-byte today's behaviour
    assert rec.calls == 3


def test_exponential_ladder_grows_and_is_capped():
    rec = Recorder(fail_times=5)
    retry_call(
        rec.func,
        retries=6,
        sleep_interval=10,
        backoff='exponential',
        max_sleep=60,
        max_total_wait=None,
        sleep=rec.sleep_sync,
    )

    # base ladder: 10, 20, 40, 60, 60 — jitter keeps each in [d/2, d)
    for expected_base, actual in zip([10, 20, 40, 60, 60], rec.delays):
        assert expected_base / 2 <= actual <= expected_base


def test_retry_after_header_is_honored_as_floor():
    exc_factory = lambda: RateLimitError('limited', response=_resp(429, headers={'retry-after': '120'}), body=None)  # noqa: E731
    rec = Recorder(fail_times=2, exc_factory=exc_factory)
    retry_call(rec.func, retries=3, sleep_interval=10, backoff='exponential', sleep=rec.sleep_sync)

    # both waits must be >= the gateway's ask, regardless of the exponential ladder
    assert all(d >= 120 for d in rec.delays)


def test_max_total_wait_gives_up_fail_closed():
    rec = Recorder(fail_times=99)
    with pytest.raises(Exception):
        retry_call(
            rec.func,
            retries=20,
            sleep_interval=10,
            backoff='exponential',
            max_sleep=60,
            max_total_wait=100,
            sleep=rec.sleep_sync,
        )
    # gave up well before exhausting the ladder
    assert rec.calls < 20


def test_non_retryable_raises_immediately_without_sleep():
    exc_factory = lambda: AuthenticationError('bad key', response=_resp(401), body=None)  # noqa: E731
    rec = Recorder(fail_times=99, exc_factory=exc_factory)
    with pytest.raises(Exception):
        retry_call(
            rec.func,
            retries=5,
            sleep_interval=10,
            no_retry_exceptions=(AuthenticationError,),
            backoff='exponential',
            sleep=rec.sleep_sync,
        )

    assert rec.calls == 1 and rec.delays == []


def test_async_parity_with_sync():
    rec_sync = Recorder(fail_times=2)
    retry_call(
        rec_sync.func,
        retries=3,
        sleep_interval=10,
        backoff='exponential',
        sleep=rec_sync.sleep_sync,
    )

    rec_async = Recorder(fail_times=2)
    loop_calls = []

    async def fail_async(*args, **kwargs):
        loop_calls.append(1)
        return rec_async.func(*args, **kwargs)

    async def sleep_async(delay):
        rec_async.delays.append(delay)

    async def run():
        await async_retry_call(
            fail_async,
            retries=3,
            sleep_interval=10,
            backoff='exponential',
            sleep=sleep_async,
        )

    import asyncio

    asyncio.run(run())

    # same attempt count, same ladder shape (base values), same bounds
    assert len(loop_calls) == rec_sync.calls
    assert rec_async.calls == rec_sync.calls
    for sync_delay, async_delay in zip(rec_sync.delays, rec_async.delays):
        assert async_delay / 2 <= sync_delay <= async_delay or sync_delay / 2 <= async_delay <= sync_delay


def test_default_callers_keep_fixed_behaviour():
    # no backoff argument -> identical to current behaviour even with the new signature
    rec = Recorder(fail_times=3)
    retry_call(rec.func, retries=5, sleep_interval=5, sleep=rec.sleep_sync)
    assert rec.delays == [5, 5, 5]
