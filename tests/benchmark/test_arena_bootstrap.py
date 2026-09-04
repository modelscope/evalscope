"""Regression tests for arena bootstrap RNG isolation (issue #1697)."""

from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd
import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.general_arena import utils
from evalscope.config import TaskConfig


@pytest.fixture(autouse=True)
def restore_numpy_rng() -> Iterator[None]:
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _statistic(battles: pd.DataFrame) -> pd.Series:
    return pd.Series({'candidate': battles['value'].mean()})


def _battles() -> pd.DataFrame:
    return pd.DataFrame({'value': np.arange(50)})


def test_bootstrap_default_is_independent_of_global_rng() -> None:
    np.random.seed(42)
    first = utils.get_bootstrap_result(_battles(), _statistic, 20)
    np.random.random(1000)
    second = utils.get_bootstrap_result(_battles(), _statistic, 20)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize('seed', [42, 7, None])
def test_bootstrap_preserves_global_rng(seed: Optional[int]) -> None:
    np.random.seed(123)
    expected = np.random.random(10)
    np.random.seed(123)
    utils.get_bootstrap_result(_battles(), _statistic, 20, seed=seed)
    np.testing.assert_array_equal(np.random.random(10), expected)


def test_bootstrap_seed_controls_samples_and_rounds_vary() -> None:
    first = utils.get_bootstrap_result(_battles(), _statistic, 20, seed=7)
    repeated = utils.get_bootstrap_result(_battles(), _statistic, 20, seed=7)
    different = utils.get_bootstrap_result(_battles(), _statistic, 20, seed=8)
    pd.testing.assert_frame_equal(first, repeated)
    assert not first.equals(different)
    assert first['candidate'].nunique() > 1


@pytest.mark.parametrize('seed', [42, 7, None])
def test_adapter_uses_task_seed_for_confidence_intervals(seed: Optional[int], monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip('sklearn')
    config = TaskConfig(
        model='candidate',
        datasets=['general_arena'],
        dataset_args={'general_arena': {'extra_params': {'baseline': 'baseline'}}},
        **({} if seed == 42 else {'seed': seed}),
    )
    adapter = get_benchmark('general_arena', config)
    scores = [
        SampleScore(
            score=Score(
                metadata={
                    'battle_result': {
                        'games': [{'model_a': 'candidate', 'model_b': 'baseline', 'judgment': judgment}],
                    },
                }
            )
        )
        for judgment in ['A>B'] * 55 + ['B>A'] * 30 + ['A=B'] * 15
    ]
    original_bootstrap = utils.get_bootstrap_result
    missing = object()

    def bootstrap(
        battles: pd.DataFrame, func_compute_elo: Callable, num_round: int, **kwargs: object
    ) -> pd.DataFrame:
        assert kwargs.get('seed', missing) == seed
        return original_bootstrap(battles, func_compute_elo, num_round, **kwargs)

    monkeypatch.setattr(utils, 'get_bootstrap_result', bootstrap)
    np.random.seed(42)
    first = adapter.aggregate_scores(scores)
    assert {score.metric_name for score in first} == {'win_rate', 'win_rate_lower', 'win_rate_upper'}
    if seed is not None:
        np.random.random(1000)
        second = adapter.aggregate_scores(scores)
        assert first == second
