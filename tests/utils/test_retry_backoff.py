"""Sync and async retry behavior with recorded delays and no real waits."""
import asyncio
from typing import Any, Optional

import httpx
import pytest
from openai import AuthenticationError, RateLimitError

from evalscope.utils.function_utils import async_retry_call, retry_call


class Recorder:
    """Record attempts and sleeps while failing a specified number of times."""

    def __init__(self, fail_times: int, error: Optional[Exception] = None) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.delays: list[float] = []
        self.error = error or RateLimitError('limited', response=_resp(), body=None)

    def func(self) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return 'ok'

    async def func_async(self) -> str:
        return self.func()

    async def sleep_async(self, delay: float) -> None:
        self.delays.append(delay)

    def run(self, async_mode: bool, **kwargs: Any) -> str:
        if async_mode:
            return asyncio.run(async_retry_call(self.func_async, sleep=self.sleep_async, **kwargs))
        return retry_call(self.func, sleep=self.delays.append, **kwargs)


def _resp(retry_after: Optional[str] = None) -> httpx.Response:
    headers = {'retry-after': retry_after} if retry_after is not None else {}
    return httpx.Response(429, headers=headers, request=httpx.Request('POST', 'http://example.test/v1'))


@pytest.fixture(params=[False, True], ids=['sync', 'async'])
def async_mode(request: pytest.FixtureRequest) -> bool:
    return request.param


def test_fixed_default_has_no_total_wait_limit(async_mode: bool) -> None:
    rec = Recorder(fail_times=4)
    assert rec.run(async_mode, retries=5, sleep_interval=100) == 'ok'
    assert rec.delays == [100, 100, 100, 100]
    assert rec.calls == 5


def test_exponential_ladder_grows_and_is_capped(async_mode: bool) -> None:
    rec = Recorder(fail_times=5)
    assert rec.run(async_mode, retries=6, sleep_interval=10, backoff='exponential') == 'ok'
    assert rec.delays == [10, 20, 40, 60, 60]
    assert rec.calls == 6


@pytest.mark.parametrize('interval', [0, 10])
def test_retry_after_can_exceed_backoff_cap(async_mode: bool, interval: int) -> None:
    error = RateLimitError('limited', response=_resp('120'), body=None)
    rec = Recorder(fail_times=2, error=error)
    assert rec.run(async_mode, retries=3, sleep_interval=interval, backoff='exponential') == 'ok'
    assert rec.delays == [120, 120]
    assert rec.calls == 3


@pytest.mark.parametrize('header', [None, 'invalid', 'Wed, 21 Oct 2015 07:28:00 GMT', '-1', '5'])
def test_retry_after_does_not_shorten_exponential_wait(async_mode: bool, header: Optional[str]) -> None:
    error = RateLimitError('limited', response=_resp(header), body=None)
    rec = Recorder(fail_times=2, error=error)
    assert rec.run(async_mode, retries=3, sleep_interval=10, backoff='exponential') == 'ok'
    assert rec.delays == [10, 20]


def test_connection_error_without_response_uses_backoff(async_mode: bool) -> None:
    rec = Recorder(fail_times=2, error=ConnectionError('disconnected'))
    assert rec.run(async_mode, retries=3, sleep_interval=10, backoff='exponential') == 'ok'
    assert rec.delays == [10, 20]


def test_total_wait_budget_preserves_original_exception(async_mode: bool) -> None:
    rec = Recorder(fail_times=99)
    with pytest.raises(RateLimitError) as caught:
        rec.run(async_mode, retries=20, sleep_interval=10, backoff='exponential', max_total_wait=70)
    assert caught.value is rec.error
    assert rec.calls == 4
    assert rec.delays == [10, 20, 40]


@pytest.mark.parametrize('budget, expected_calls, expected_delays', [(100, 1, []), (240, 3, [120, 120])])
def test_budget_never_shortens_retry_after(
    async_mode: bool, budget: int, expected_calls: int, expected_delays: list[float]
) -> None:
    error = RateLimitError('limited', response=_resp('120'), body=None)
    rec = Recorder(fail_times=99, error=error)
    with pytest.raises(RateLimitError) as caught:
        rec.run(async_mode, retries=5, sleep_interval=0, backoff='exponential', max_total_wait=budget)
    assert caught.value is error
    assert rec.calls == expected_calls
    assert rec.delays == expected_delays


def test_non_retryable_error_raises_without_sleep(async_mode: bool) -> None:
    error = AuthenticationError('bad key', response=_resp(), body=None)
    rec = Recorder(fail_times=99, error=error)
    with pytest.raises(AuthenticationError) as caught:
        rec.run(
            async_mode, retries=5, sleep_interval=10, backoff='exponential',
            no_retry_exceptions=(AuthenticationError,),
        )
    assert caught.value is error
    assert rec.calls == 1
    assert rec.delays == []


@pytest.mark.parametrize('backoff', ['fixed', 'exponential'])
def test_zero_interval_without_header_retries_without_sleep(async_mode: bool, backoff: str) -> None:
    rec = Recorder(fail_times=2)
    assert rec.run(async_mode, retries=3, sleep_interval=0, backoff=backoff) == 'ok'
    assert rec.calls == 3
    assert rec.delays == []


def test_fixed_mode_ignores_retry_after(async_mode: bool) -> None:
    error = RateLimitError('limited', response=_resp('120'), body=None)
    rec = Recorder(fail_times=2, error=error)
    assert rec.run(async_mode, retries=3, sleep_interval=10) == 'ok'
    assert rec.delays == [10, 10]
