"""Fractional credit cannot be converted into a count of successful generations."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.metrics.aggregators import MeanPassAtK, MeanPassHatK


@pytest.mark.parametrize('cls', [MeanPassAtK, MeanPassHatK])
@pytest.mark.parametrize('invalid', [0.5, -1, 2, float('nan'), float('inf'), -float('inf')])
def test_nonbinary_metric_keeps_mean_but_has_no_pass_probability(cls: type, invalid: float) -> None:
    scores = [SampleScore(sample_id=i, group_id='g', generation_index=i,
                          score=Score(value={'partial': invalid, 'acc': value}))
              for i, value in enumerate([0, 1])]
    result = cls()(scores)
    assert any(a.metric_name == 'partial' and a.aggregation == 'mean' for a in result)
    assert not any(a.metric_name == 'partial' and a.dimensions.get('k') for a in result)
    binary = [a for a in result if a.metric_name == 'accuracy' and a.dimensions.get('k') == 2]
    assert len(binary) == 1
    assert binary[0].score == (1.0 if cls is MeanPassAtK else 0.0)


@pytest.mark.parametrize('cls', [MeanPassAtK, MeanPassHatK])
def test_partial_credit_disables_whole_metric_not_only_one_question(cls: type) -> None:
    scores = [SampleScore(sample_id=i, group_id=str(i // 2), generation_index=i % 2,
                          score=Score(value={'acc': value})) for i, value in enumerate([1, 1, 0.5, 0.5])]
    result = cls()(scores)
    assert len(result) == 1
    assert result[0].score == 0.75
    assert result[0].num == 4


@pytest.mark.parametrize('aggregation', ['mean_and_pass_at_k', 'mean_and_pass_hat_k'])
def test_native_multiple_choice_cache_and_concurrent_aggregation(aggregation: str) -> None:
    cfg = TaskConfig(model='mock', eval_type='mock_llm', datasets=['general_mcq'], repeats=2,
                     dataset_args={'general_mcq': {'aggregation': aggregation,
                                                   'extra_params': {'multiple_correct': True}}})
    adapter = get_benchmark('general_mcq', cfg)
    scores = []
    for i in range(2):
        sample = Sample(input='Choose both.', choices=['first', 'second'], target=['A', 'B'], id=i, group_id=0)
        state = TaskState(model='mock', sample=sample, completed=True,
                          output=ModelOutput.from_content('mock', 'A'))
        score = adapter.calculate_metrics(state)
        assert score.score.main_value == 0.5
        scores.append(SampleScore.model_validate_json(score.model_dump_json()))

    def aggregate(reverse: bool) -> list:
        return adapter.aggregate_scores(list(reversed(scores)) if reverse else scores)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(aggregate, [False, True] * 8))
    for result in results:
        assert len(result) == 1
        assert result[0].aggregation == 'mean'
        assert result[0].score == 0.5
