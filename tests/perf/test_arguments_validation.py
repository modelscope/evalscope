"""Unit tests for perf ``Arguments`` validation guards.

Covers three previously missing validations:
- ``--open-loop --multi-turn``: used to be accepted silently; open-loop then
  forced ``parallel=[-1]`` so the multi-turn strategy spawned zero workers and
  the run produced no requests at all.
- ``--parallel <= 0`` (closed loop): used to reach the strategy and crash
  opaquely ('Set of Tasks/Futures is empty' / 'Semaphore initial value must be
  >= 0').
- ``--log-every-n-query 0``: used to cause a ZeroDivisionError in the metrics
  consumer's modulo; now coerced to 1 like the other count-type knobs.
"""
import pytest

from evalscope.perf.arguments import Arguments


def _args(**kwargs) -> Arguments:
    return Arguments(model='test-model', url='http://localhost:8080/v1/chat/completions', **kwargs)


class TestOpenLoopMultiTurnRejected:

    def test_open_loop_with_multi_turn_raises(self):
        with pytest.raises(ValueError, match='not supported in open-loop'):
            _args(open_loop=True, multi_turn=True, rate=1.0, number=2)

    def test_multi_turn_closed_loop_still_accepted(self):
        args = _args(multi_turn=True, parallel=2, number=2)
        assert args.multi_turn is True

    def test_open_loop_single_turn_still_accepted(self):
        args = _args(open_loop=True, rate=1.0, number=2)
        assert args.parallel == [-1]  # unbounded concurrency marker stays intact


class TestParallelPositivity:

    @pytest.mark.parametrize('parallel', [0, -1, -5])
    def test_non_positive_parallel_rejected(self, parallel):
        with pytest.raises(ValueError, match='--parallel values must be > 0'):
            _args(parallel=parallel)

    @pytest.mark.parametrize('parallel', [1, 4, [8]])
    def test_positive_parallel_accepted(self, parallel):
        assert _args(parallel=parallel).parallel == ([parallel] if isinstance(parallel, int) else parallel)

    def test_positive_parallel_sweep_accepted(self):
        assert _args(parallel=[1, 2], number=[10, 10]).parallel == [1, 2]

    def test_zero_inside_sweep_rejected(self):
        with pytest.raises(ValueError, match='--parallel values must be > 0'):
            _args(parallel=[1, 0, 2], number=[1, 1, 1])

    def test_open_loop_parallel_marker_not_affected(self):
        # Open-loop mode force-sets parallel=[-1] internally; the positivity
        # rule applies only to closed-loop sweeps.
        args = _args(open_loop=True, rate=1.0, number=2)
        assert args.parallel == [-1]


class TestLogEveryNQuery:

    @pytest.mark.parametrize('value', [0, -3])
    def test_non_positive_is_coerced_to_one(self, value):
        assert _args(log_every_n_query=value).log_every_n_query == 1

    def test_positive_value_preserved(self):
        assert _args(log_every_n_query=50).log_every_n_query == 50

    def test_default_preserved(self):
        assert _args().log_every_n_query == 100
