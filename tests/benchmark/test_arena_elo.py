"""Win rates from the arena Bradley-Terry fit, including one-sided battles."""

import pandas as pd
import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.arena_hard import utils as arena_hard_utils
from evalscope.benchmarks.general_arena import utils as general_arena_utils
from evalscope.config import TaskConfig

pytest.importorskip('sklearn')

BASELINE = 'gpt4-0314'


def _battles(*records):
    """Battles against the baseline: (model, wins, losses, ties)."""
    rows = []
    for model, wins, losses, ties in records:
        rows += [{'model_a': model, 'model_b': BASELINE, 'winner': 'model_a'}] * wins
        rows += [{'model_a': model, 'model_b': BASELINE, 'winner': 'model_b'}] * losses
        rows += [{'model_a': model, 'model_b': BASELINE, 'winner': 'tie'}] * ties
    return pd.DataFrame(rows)


def _win_rates(utils, battles):
    kwargs = {'baseline_model': BASELINE} if utils is general_arena_utils else {}
    elo = utils.compute_mle_elo(battles, **kwargs)
    stats = pd.DataFrame({'model': list(elo.index), 'score': list(elo.values)})
    return utils.get_win_rate_column(stats, 'score', BASELINE).to_dict()


UTILS = pytest.mark.parametrize('utils', [arena_hard_utils, general_arena_utils], ids=['arena_hard', 'general_arena'])


@UTILS
@pytest.mark.parametrize('wins, losses, ties, expected', [(7, 3, 0, 0.7), (19, 1, 0, 0.95), (5, 3, 2, 0.6)])
def test_win_rate_matches_the_record_against_the_baseline(utils, wins, losses, ties, expected):
    """With one candidate the fitted win rate is the share of games won, a tie counting half."""
    assert _win_rates(utils, _battles(('m1', wins, losses, ties)))['m1'] == pytest.approx(expected, abs=1e-4)


@UTILS
def test_a_model_that_wins_every_game_has_win_rate_one(utils):
    assert _win_rates(utils, _battles(('m1', 10, 0, 0)))['m1'] == 1.0


@UTILS
def test_a_model_that_loses_every_game_has_win_rate_zero(utils):
    assert _win_rates(utils, _battles(('m1', 0, 10, 0)))['m1'] == 0.0


def test_every_sweeping_model_is_scored_not_only_the_first():
    rates = _win_rates(general_arena_utils, _battles(('m1', 10, 0, 0), ('m2', 10, 0, 0)))

    assert rates['m1'] == 1.0
    assert rates['m2'] == 1.0


@UTILS
def test_a_sweep_does_not_rank_below_a_model_that_dropped_a_game(utils):
    """Scored in separate runs, as two candidates would be."""
    sweep = _win_rates(utils, _battles(('m1', 10, 0, 0)))['m1']
    one_loss = _win_rates(utils, _battles(('m1', 19, 1, 0)))['m1']

    assert sweep > one_loss


def test_bootstrap_interval_of_a_sweep_is_one():
    battles = _battles(('m1', 10, 0, 0))
    boot = general_arena_utils.get_bootstrap_result(
        battles, general_arena_utils.compute_mle_elo, 20, baseline_model=BASELINE, seed=0
    )

    stats = pd.DataFrame({'model': list(boot.columns), 'lower': [boot[m].quantile(0.025) for m in boot.columns]})

    assert general_arena_utils.get_win_rate_column(stats, 'lower', BASELINE)['m1'] == 1.0


def test_general_arena_num_counts_only_samples_with_a_battle():
    config = TaskConfig(
        model='candidate',
        datasets=['general_arena'],
        dataset_args={'general_arena': {'extra_params': {'baseline': 'baseline'}}},
    )
    adapter = get_benchmark('general_arena', config)
    games = [{'model_a': 'candidate', 'model_b': 'baseline', 'judgment': j} for j in ['A>B'] * 6 + ['B>A'] * 4]
    scores = [SampleScore(score=Score(metadata={'battle_result': {'games': [game]}})) for game in games]
    scores += [SampleScore(score=Score(metadata={})) for _ in range(5)]  # judge verdicts unusable

    agg = adapter.aggregate_scores(scores)

    assert {score.num for score in agg} == {10}


def _arena_hard_sample(first, second):
    """One arena_hard sample: the baseline is model_a; game 2 is played with the answers swapped."""
    games = [{'score': first}, {'score': second}]
    return SampleScore(score=Score(metadata={'battle_result': {'model_a': BASELINE, 'model_b': 'test_model', 'games': games}}))


def _general_arena_sample(model, first, second):
    games = [
        {'model_a': model, 'model_b': 'baseline', 'judgment': first},
        {'model_a': 'baseline', 'model_b': model, 'judgment': second},
    ]
    return SampleScore(score=Score(metadata={'battle_result': {'games': games}}))


@pytest.mark.parametrize('first, second, expected', [('B>A', 'A>B', 1.0), ('A>B', 'B>A', 0.0)])
def test_arena_hard_reports_a_one_sided_run_as_won_or_lost(first, second, expected):
    adapter = get_benchmark('arena_hard', TaskConfig(model='m', datasets=['arena_hard']))

    agg = adapter.aggregate_scores([_arena_hard_sample(first, second)] * 5)

    assert [(score.metric_name, score.score) for score in agg] == [('win_rate', expected)]


def test_general_arena_reports_every_sweeping_model_and_its_interval():
    config = TaskConfig(
        model='candidate',
        datasets=['general_arena'],
        dataset_args={'general_arena': {'extra_params': {'baseline': 'baseline'}}},
    )
    adapter = get_benchmark('general_arena', config)
    scores = [_general_arena_sample(model, 'A>B', 'B>A') for model in ('m1', 'm2') for _ in range(5)]

    agg = adapter.aggregate_scores(scores)

    assert {(score.metric_name, score.score) for score in agg} == {
        ('win_rate', 1.0),
        ('win_rate_lower', 1.0),
        ('win_rate_upper', 1.0),
    }
    assert len(agg) == 6  # three metrics for each of the two models


def test_scoring_change_raises_the_evaluation_version():
    assert get_benchmark('arena_hard', TaskConfig(model='m', datasets=['arena_hard'])).benchmark_meta.evaluation_version == 'v1.1'
    config = TaskConfig(model='m', datasets=['general_arena'], dataset_args={'general_arena': {'extra_params': {'baseline': 'b'}}})
    assert get_benchmark('general_arena', config).benchmark_meta.evaluation_version == 'v1.3'
