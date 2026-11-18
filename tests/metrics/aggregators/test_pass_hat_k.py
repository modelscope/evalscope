"""Preserve pass^k estimates while bounding repeated input traversal."""

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import pytest

from evalscope import TaskConfig, run_task
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.model.model import ModelCache
from evalscope.metrics.aggregators import Mean, MeanPassHatK, aggregators
from evalscope.models.mockllm import MockLLM


def reference_pass_hat(scores: list[SampleScore]) -> list[AggScore]:
    """Rescan each eligible prefix using the pre-optimization estimator."""
    result = Mean()(scores)
    metrics = dict.fromkeys(key for sample in scores for key in sample.score.value)
    for metric in metrics:
        if any(s.score.value[metric] not in (0, 1) for s in scores if metric in s.score.value):
            continue
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
                outcomes = [float(s.score.value[metric]) for s in attempts.values() if s is not None]
                values.append(math.comb(int(sum(outcomes)), k) / math.comb(len(outcomes), k))
                ids.append(group)
            if values:
                result.append(AggScore(score=sum(values) / len(values), metric_name=metric, aggregation='pass_hat_k',
                                       dimensions={'k': k}, num=len(values), ids=ids,
                                       metadata={'eligible': len(values), 'total': len(groups),
                                                 'coverage': len(values) / len(groups),
                                                 'excluded': len(groups) - len(values)}))
    return result


def _dump(scores: list[AggScore]) -> list[str]:
    return [score.model_dump_json() for score in scores]


def _sample(value: float | None, position: int | None, group: Any = 'g', sample_id: Any = 0) -> SampleScore:
    return SampleScore(sample_id=sample_id, group_id=group, generation_index=position,
                       score=Score(value={} if value is None else {'acc': value}))


def test_estimates_use_all_available_attempts_even_after_a_gap() -> None:
    samples = [_sample(value, i) for i, value in enumerate([1, 1, None, 0])]
    result = MeanPassHatK()(samples)
    estimates = [a for a in result if a.aggregation == 'pass_hat_k']
    assert [(a.dimensions['k'], a.score) for a in estimates] == [(1, 2 / 3), (2, 1 / 3)]
    assert _dump(result) == _dump(reference_pass_hat(samples))


@pytest.mark.parametrize('seed', range(40))
def test_randomized_outputs_match_reference(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(10):
        scores = []
        for group in range(rng.randrange(1, 7)):
            for position in range(rng.randrange(1, 24)):
                if seed % 2 and rng.random() < 0.08:
                    continue
                values = {key: rng.choice([0.0, 1.0]) for key in ['acc', 'other']
                          if not seed % 2 or rng.random() > 0.18}
                scores.append(SampleScore(sample_id=f'{group}-{position}', group_id=group,
                                          generation_index=position if seed % 3 else None,
                                          score=Score(value=values)))
                if seed % 5 == 0:
                    scores.append(scores[-1].model_copy(update={'score': Score(value={'acc': 0.0})}))
        rng.shuffle(scores)
        before = [s.model_dump_json() for s in scores]
        assert _dump(MeanPassHatK()(scores)) == _dump(reference_pass_hat(scores))
        assert [s.model_dump_json() for s in scores] == before


@pytest.mark.parametrize('rows', [[], [('sample', None, None)], [(0, 0, 0), (0, 0, 0)],
                                 [(0, None, 0), (0, None, 1)], [(0, 0, -1), (0, 0, 0)],
                                 [(0, 0, -2)], [(0, 0, 0), (1, 0, 2), (2, 1, 0), (3, 1, 1)]])
def test_identity_fallbacks_duplicates_and_sparse_positions(rows: list[tuple]) -> None:
    scores = [_sample(float(i % 2), position, gid, sid) for i, (sid, gid, position) in enumerate(rows)]
    assert _dump(MeanPassHatK()(scores)) == _dump(reference_pass_hat(scores))


@pytest.mark.parametrize('value', [0, 1, True, False, 0.5, -1, 2, float('nan'), float('inf')])
def test_constant_and_nonbinary_outcomes(value: float) -> None:
    scores = [_sample(value, i) for i in range(4)]
    assert _dump(MeanPassHatK()(scores)) == _dump(reference_pass_hat(scores))


class CountingSample(SampleScore):
    score_reads: ClassVar[int] = 0

    def __getattribute__(self, name: str) -> Any:
        if name == 'score':
            type(self).score_reads += 1
        return super().__getattribute__(name)


@pytest.mark.parametrize('repeats', [32, 128])
def test_each_score_is_read_a_bounded_number_of_times(repeats: int) -> None:
    samples = [CountingSample(sample_id=i, group_id=0, generation_index=i, score=Score(value={'acc': float(i % 2)}))
               for i in range(repeats)]
    CountingSample.score_reads = 0
    assert len(MeanPassHatK()(samples)) == repeats + 1
    assert CountingSample.score_reads <= 6 * repeats


def test_groups_without_an_eligible_prefix_do_not_rescan_outcomes() -> None:
    samples = [CountingSample(sample_id=i, group_id=0, generation_index=i,
                              score=Score(value={'acc': 1} if i else {})) for i in range(128)]
    CountingSample.score_reads = 0
    assert len(MeanPassHatK()(samples)) == 1
    assert CountingSample.score_reads <= 5 * len(samples)


@pytest.mark.parametrize('repeats', [32, 128])
def test_prefix_eligibility_is_checked_in_linear_work(repeats: int, monkeypatch: pytest.MonkeyPatch) -> None:
    class CountingAttempts(dict):
        reads = 0

        def get(self, key: Any, default: Any = None) -> Any:
            type(self).reads += 1
            return super().get(key, default)

    collect = aggregators.collect_planned_attempts
    monkeypatch.setattr(aggregators, 'collect_planned_attempts', lambda *args: {
        key: CountingAttempts(value) for key, value in collect(*args).items()})
    assert len(MeanPassHatK()([_sample(1, i) for i in range(repeats)])) == repeats + 1
    assert CountingAttempts.reads <= 2 * repeats


def test_one_aggregator_can_be_reused_concurrently() -> None:
    batches = [[_sample(float(i % (group + 2) == 0), i, group) for i in range(64)] for group in range(8)]
    expected = [_dump(reference_pass_hat(batch)) for batch in batches]
    aggregator = MeanPassHatK()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: _dump(aggregator(batches[i % 8])), range(64)))
    assert results == expected * 8


def test_native_repeat_pipeline_report_and_cached_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ModelCache, '_models', {})
    monkeypatch.setattr(MockLLM, 'default_output', 'A')
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'tiny.jsonl').write_text(''.join(json.dumps({'question': f'Question {i}', 'A': 'first', 'B': 'second',
                                                        'answer': answer}) + '\n'
                                             for i, answer in enumerate(['A', 'B'])))
    output = tmp_path / 'output'
    config = TaskConfig(model='pass-hat-test', eval_type='mock_llm', datasets=['general_mcq'], repeats=8,
                        dataset_args={'general_mcq': {'local_path': str(data), 'subset_list': ['tiny'],
                                                      'aggregation': 'mean_and_pass_hat_k', 'few_shot_num': 0}},
                        eval_batch_size=4, work_dir=str(output), no_timestamp=True)
    run_task(config)
    review_path = next((output / 'reviews').rglob('*.jsonl'))
    reviews = [SampleScore.model_validate(json.loads(row)['sample_score']) for row in review_path.read_text().splitlines()]
    assert len(reviews) == 16
    assert _dump(MeanPassHatK()(reviews)) == _dump(reference_pass_hat(reviews))
    report_path = next((output / 'reports').rglob('general_mcq.json'))
    report = json.loads(report_path.read_text())
    estimates = [m for m in report['metrics'] if m['identity']['aggregation'] == 'pass_hat_k']
    assert len(estimates) == 8
    assert all(m['score'] == 0.5 and m['num'] == 2 for m in estimates)
    assert (output / 'reports' / 'report.html').is_file()
    predictions = {path: path.read_bytes() for path in (output / 'predictions').rglob('*.jsonl')}

    def unexpected_generate(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('Cached predictions must not be regenerated.')

    monkeypatch.setattr(MockLLM, 'generate', unexpected_generate)
    config.use_cache = str(output)
    run_task(config)
    assert json.loads(report_path.read_text())['metrics'] == report['metrics']
    config.rerun_review = True
    run_task(config)
    assert json.loads(report_path.read_text())['metrics'] == report['metrics']
    assert all(path.read_bytes() == content for path, content in predictions.items())
