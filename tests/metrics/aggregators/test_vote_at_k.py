"""Lock vote@k semantics and linear input traversal independently of wall-clock speed."""

import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import pytest

from evalscope import TaskConfig, run_task
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.model.model import ModelCache
from evalscope.metrics.aggregators import Mean, MeanVoteAtK
from evalscope.models.mockllm import MockLLM


def reference_vote(scores: list[SampleScore]) -> list[AggScore]:
    """Deliberately rescan each prefix, retaining the pre-optimization definition."""
    result = Mean()(scores)
    metrics = dict.fromkeys(key for sample in scores for key in sample.score.value)
    for metric in metrics:
        groups = {}
        for sample in scores:
            group = sample.group_id if sample.group_id is not None else sample.sample_id
            attempts = groups.setdefault(group, {})
            position = sample.generation_index if sample.generation_index is not None else len(attempts)
            attempts[position] = sample if metric in sample.score.value else None
        max_k = min((max(attempts, default=-1) + 1 for attempts in groups.values()), default=0)
        for k in range(1, max_k + 1):
            values, ids = [], []
            for group, attempts in groups.items():
                if any(attempts.get(i) is None for i in range(k)):
                    continue
                prefix = [attempts[i].score for i in range(k)]
                counts = Counter(s.extracted_prediction for s in prefix
                                 if s.extracted_prediction is not None and s.extracted_prediction.strip())
                winner = counts.most_common(1)[0][0] if counts else None
                correct = winner is not None and any(s.extracted_prediction == winner and s.value[metric] == 1.0
                                                    for s in prefix)
                values.append(float(correct))
                ids.append(group)
            if values:
                result.append(AggScore(score=sum(values) / len(values), metric_name=metric, aggregation='vote_at_k',
                                       dimensions={'k': k}, num=len(values), ids=ids,
                                       metadata={'eligible': len(values), 'total': len(groups),
                                                 'coverage': len(values) / len(groups),
                                                 'excluded': len(groups) - len(values)}))
    return result


def _dump(scores: list[AggScore]) -> list[dict]:
    return [score.model_dump() for score in scores]


def _sample(prediction: str | None, correct: float, position: int, group: Any = 'g') -> SampleScore:
    return SampleScore(sample_id=f'{group}-{position}', group_id=group, generation_index=position,
                       score=Score(value={'acc': correct}, extracted_prediction=prediction))


def test_earlier_prediction_recovers_a_tie_after_losing_the_lead() -> None:
    scores = [_sample(p, float(p == 'a'), i) for i, p in enumerate(['a', 'b', 'b', 'a', 'b'])]
    votes = [a.score for a in MeanVoteAtK()(scores) if a.aggregation == 'vote_at_k']
    assert votes == [1.0, 1.0, 0.0, 1.0, 0.0]
    assert _dump(MeanVoteAtK()(scores)) == _dump(reference_vote(scores))


def test_later_correct_observation_updates_an_existing_winner() -> None:
    scores = [_sample('same', value, i) for i, value in enumerate([0.0, 0.5, 1.0])]
    assert [a.score for a in MeanVoteAtK()(scores) if a.aggregation == 'vote_at_k'] == [0.0, 0.0, 1.0]


@pytest.mark.parametrize('seed', range(40))
def test_randomized_outputs_match_reference(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(10):
        scores = []
        for group in range(rng.randrange(1, 7)):
            for position in range(rng.randrange(1, 24)):
                if rng.random() < 0.08:
                    continue
                values = {key: rng.choice([0.0, 0.5, 1.0, 2.0]) for key in ['acc', 'other'] if rng.random() > 0.18}
                scores.append(SampleScore(
                    sample_id=f'{group}-{position}', group_id=group,
                    generation_index=position if seed % 3 else None,
                    score=Score(value=values, extracted_prediction=rng.choice(['a', 'b', 'c', ' a', '', ' ', None]))))
        rng.shuffle(scores)
        before = [s.model_dump() for s in scores]
        assert _dump(MeanVoteAtK()(scores)) == _dump(reference_vote(scores))
        assert [s.model_dump() for s in scores] == before


@pytest.mark.parametrize('rows', [[], [('sample', None, None)], [(0, 0, 0), (0, 0, 0)],
                                 [(0, None, 0), (0, None, 1)], [(0, 0, -1), (0, 0, 0)],
                                 [(0, 0, 0), (1, 0, 2), (2, 1, 0), (3, 1, 1)]])
def test_identity_fallbacks_duplicates_and_sparse_positions(rows: list[tuple]) -> None:
    scores = [SampleScore(sample_id=sid, group_id=gid, generation_index=position,
                          score=Score(value={'acc': float(i % 2)}, extracted_prediction=str(i % 2)))
              for i, (sid, gid, position) in enumerate(rows)]
    assert _dump(MeanVoteAtK()(scores)) == _dump(reference_vote(scores))


class CountingSample(SampleScore):
    score_reads: ClassVar[int] = 0

    def __getattribute__(self, name: str) -> Any:
        if name == 'score':
            type(self).score_reads += 1
        return super().__getattribute__(name)


@pytest.mark.parametrize('repeats', [32, 128])
def test_each_attempt_is_read_a_bounded_number_of_times(repeats: int) -> None:
    samples = [CountingSample(sample_id=i, group_id=0, generation_index=i,
                              score=Score(value={'acc': float(i % 2)}, extracted_prediction=str(i)))
               for i in range(repeats)]
    CountingSample.score_reads = 0
    result = MeanVoteAtK()(samples)
    assert len(result) == repeats + 1
    assert CountingSample.score_reads <= 6 * repeats


def test_one_aggregator_can_be_reused_concurrently() -> None:
    batches = [[_sample(str(i % (2 + group)), float(i % 2), i, group) for i in range(64)] for group in range(8)]
    expected = [_dump(reference_vote(batch)) for batch in batches]
    aggregator = MeanVoteAtK()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: _dump(aggregator(batches[i % 8])), range(64)))
    assert results == expected * 8


def test_native_repeat_pipeline_and_cached_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ModelCache, '_models', {})
    monkeypatch.setattr(MockLLM, 'default_output', 'A')
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'tiny.jsonl').write_text(''.join(json.dumps({'question': f'Question {i}', 'A': 'first', 'B': 'second',
                                                        'answer': answer}) + '\n'
                                             for i, answer in enumerate(['A', 'B'])))
    output = tmp_path / 'output'
    config = TaskConfig(model='vote-test', eval_type='mock_llm', datasets=['general_mcq'], repeats=8,
                        dataset_args={'general_mcq': {'local_path': str(data), 'subset_list': ['tiny'],
                                                      'aggregation': 'mean_and_vote_at_k', 'few_shot_num': 0,
                                                      'primary_metric': MetricSelector(name='accuracy', aggregation='mean')}},
                        eval_batch_size=4, work_dir=str(output), no_timestamp=True)
    run_task(config)
    review_path = next((output / 'reviews').rglob('*.jsonl'))
    reviews = [SampleScore.model_validate(json.loads(row)['sample_score']) for row in review_path.read_text().splitlines()]
    assert len(reviews) == 16
    assert _dump(MeanVoteAtK()(reviews)) == _dump(reference_vote(reviews))
    report_path = next((output / 'reports').rglob('general_mcq.json'))
    report = json.loads(report_path.read_text())
    votes = [m for m in report['metrics'] if m['identity']['aggregation'] == 'vote_at_k']
    assert len(votes) == 8
    assert all(m['score'] == 0.5 and m['num'] == 2 for m in votes)

    def unexpected_generate(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('Cached predictions must not be regenerated.')

    monkeypatch.setattr(MockLLM, 'generate', unexpected_generate)
    config.use_cache = str(output)
    config.rerun_review = True
    run_task(config)
    assert json.loads(report_path.read_text())['metrics'] == report['metrics']
