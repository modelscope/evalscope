"""Tests for the multi-turn ``startup_ramp_seconds`` mechanism."""
import argparse
import asyncio
import inspect
import numpy as np
import pytest
from unittest.mock import patch

from evalscope.perf.arguments import Arguments, add_argument
from evalscope.perf.core.strategies import multi_turn as mt
from evalscope.perf.core.strategies.multi_turn import MultiTurnStrategy

# --- Field defaults, CLI, validators ---


_RAMP_FIELDS = [
    ('startup_ramp_seconds', '--startup-ramp-seconds', None),
    ('warmup_ramp_min_conversations', '--warmup-ramp-min-conversations', 3),
    ('benchmark_ramp_min_parallel', '--benchmark-ramp-min-parallel', 3),
]


@pytest.mark.parametrize('field,cli,default', _RAMP_FIELDS)
def test_field_default_and_cli(field, cli, default):
    f = Arguments.model_fields[field]
    assert f.default is default
    p = argparse.ArgumentParser()
    add_argument(p)
    actions = [a for a in p._actions if cli in a.option_strings]
    assert actions and actions[0].default is default


def test_negative_threshold_rejected():
    with pytest.raises(Exception) as ei:
        Arguments.model_validate({
            'model': 'm', 'url': 'http://x/v1/chat/completions',
            'dataset': 'openqa', 'parallel': 4, 'number': 10,
            'multi_turn': True,
            'warmup_ramp_min_conversations': -1,
        })
    assert 'warmup-ramp-min-conversations' in str(ei.value)


def test_threshold_zero_disables_guardrail():
    args = Arguments.model_validate({
        'model': 'm', 'url': 'http://x/v1/chat/completions',
        'dataset': 'openqa', 'parallel': 4, 'number': 10,
        'multi_turn': True,
        'startup_ramp_seconds': 5.0,
        'warmup_ramp_min_conversations': 0,
        'benchmark_ramp_min_parallel': 0,
    })
    assert args.warmup_ramp_min_conversations == 0
    assert args.benchmark_ramp_min_parallel == 0


# --- No module-level constants; strategy reads via self.args ---


def test_thresholds_live_on_arguments():
    assert not hasattr(mt, '_WARMUP_RAMP_MIN_CONVERSATIONS')
    assert not hasattr(mt, '_BENCHMARK_RAMP_MIN_PARALLEL')
    src = inspect.getsource(MultiTurnStrategy.run)
    assert 'self.args.warmup_ramp_min_conversations' in src
    assert 'self.args.benchmark_ramp_min_parallel' in src


# --- Schedule math ---


@pytest.mark.parametrize('seed', [0, 1, 42, 999])
def test_rescale_locks_last_worker_to_ramp(seed):
    n, ramp_s = 32, 12.0
    np.random.seed(seed)
    offsets = np.cumsum(np.random.exponential(ramp_s / (n - 1), size=n - 1))
    if offsets[-1] > 0:
        offsets *= ramp_s / offsets[-1]
    assert offsets[-1] == pytest.approx(ramp_s, abs=1e-9)


def test_rescale_preserves_gap_ratios():
    n, ramp_s = 50, 5.0
    np.random.seed(7)
    offsets = np.cumsum(np.random.exponential(ramp_s / (n - 1), size=n - 1))
    ratios_before = np.diff(offsets) / offsets[-1]
    np.testing.assert_allclose(
        ratios_before, np.diff(offsets * ramp_s / offsets[-1]) / ramp_s,
    )


# --- Spawn loop with mocked clock ---


@pytest.fixture
def fake_clock(monkeypatch):
    state = {'now': 1000.0}
    monkeypatch.setattr(mt.time, 'perf_counter', lambda: state['now'])

    async def sleep(s):
        state['now'] += s
    monkeypatch.setattr(mt.asyncio, 'sleep', sleep)
    return state


@pytest.fixture
def noop_worker(monkeypatch):
    async def noop(self, worker_id):  # noqa: ARG001
        return None
    monkeypatch.setattr(MultiTurnStrategy, '_worker', noop)


@pytest.fixture
def capture_create_task(monkeypatch):
    captured = []
    real = asyncio.create_task
    monkeypatch.setattr(
        mt.asyncio, 'create_task',
        lambda coro, **kw: captured.append(mt.time.perf_counter()) or real(coro, **kw),
    )
    return captured


def _strategy(parallel, ramp_s):
    args = Arguments.model_validate({
        'model': 'm', 'url': 'http://x/v1/chat/completions',
        'dataset': 'openqa', 'parallel': parallel, 'number': 10,
        'multi_turn': True, 'warmup_num': 0,
        'startup_ramp_seconds': ramp_s,
    })
    if isinstance(args.parallel, list):
        args.parallel = args.parallel[0]
    if isinstance(args.number, list):
        args.number = args.number[0]
    return MultiTurnStrategy(args, None, None, asyncio.Queue(), [])


async def _drive(strategy, apply_ramp):
    """Run ``_spawn_workers`` and clean up no-op tasks."""
    workers = await strategy._spawn_workers(apply_startup_ramp=apply_ramp)
    for w in workers:
        w.cancel()
    if workers:
        await asyncio.gather(*workers, return_exceptions=True)


@pytest.mark.parametrize(
    'parallel,ramp_s,apply_ramp,expected_advance',
    [
        (20, 10.0, True, 10.0),     # ramp on advances exactly ramp_s
        (1, 5.0, True, 0.0),        # parallel=1 short-circuit
        (16, None, True, 0.0),      # ramp disabled (None)
        (8, 0.0, True, 0.0),        # ramp disabled (0)
        (8, 4.0, False, 0.0),       # apply_ramp=False short-circuits
    ],
)
def test_spawn_loop_semantics(
    parallel, ramp_s, apply_ramp, expected_advance,
    fake_clock, noop_worker, capture_create_task,
):
    asyncio.run(_drive(_strategy(parallel, ramp_s), apply_ramp))

    if parallel > 0:
        assert len(capture_create_task) == parallel
    assert fake_clock['now'] - 1000.0 == pytest.approx(expected_advance, abs=1e-9)


def test_ramp_first_worker_immediate_last_at_exactly_ramp(
    fake_clock, noop_worker, capture_create_task,
):
    n, ramp_s = 20, 10.0
    asyncio.run(_drive(_strategy(n, ramp_s), apply_ramp=True))

    assert capture_create_task[0] == pytest.approx(1000.0, abs=1e-9)
    assert capture_create_task[-1] == pytest.approx(1000.0 + ramp_s, abs=1e-9)
    assert (np.diff(capture_create_task) > 0).all()


def test_spawn_zero_parallel_returns_empty(
    fake_clock, noop_worker, capture_create_task,
):
    s = _strategy(parallel=0, ramp_s=10.0)
    asyncio.run(_drive(s, apply_ramp=True))
    assert capture_create_task == []
    assert fake_clock['now'] == 1000.0


def test_spawn_does_not_sleep_past_phase_deadline(
    fake_clock, noop_worker, capture_create_task,
):
    n, ramp_s = 20, 10.0
    deadline = 1000.0 + 5.0  # mid-ramp

    async def drive_with_deadline():
        s = _strategy(n, ramp_s)
        s._phase_deadline = deadline
        workers = await s._spawn_workers(apply_startup_ramp=True)
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    asyncio.run(drive_with_deadline())

    assert fake_clock['now'] <= deadline
    assert len(capture_create_task) == n


# --- Lifetime: ramp helper called exactly once per phase ---


def test_spawn_workers_called_exactly_once_per_phase(
    fake_clock, noop_worker, capture_create_task,
):
    s = _strategy(parallel=100, ramp_s=30.0)
    spawn_calls = {'n': 0}
    real_spawn = s._spawn_workers

    async def counting(apply_startup_ramp):
        spawn_calls['n'] += 1
        return await real_spawn(apply_startup_ramp)

    with patch.object(MultiTurnStrategy, '_spawn_workers', side_effect=counting):
        async def drive():
            s._phase_counter = 0
            s._phase_budget = 100
            s._phase_is_warmup = False
            s._phase_deadline = None
            await s._run_phase(
                budget=100, is_warmup=False, deadline=None,
                apply_startup_ramp=True,
            )

        asyncio.run(drive())

    assert spawn_calls['n'] == 1
    assert len(capture_create_task) == 100


# --- run() gate skip-log gating ---


def test_run_skip_logs_gated_by_ramp_configured():
    src = inspect.getsource(MultiTurnStrategy.run)
    assert 'if not warmup_apply_ramp and self.args.startup_ramp_seconds:' in src
    assert 'if not benchmark_apply_ramp and self.args.startup_ramp_seconds:' in src
