import json
from fractions import Fraction
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evalscope import TaskConfig, run_task
from evalscope.api.metric import MetricDirection, MetricKind, SampleScore, Score
from evalscope.api.registry import get_aggregation, get_benchmark
from evalscope.constants import ScoreStatus
from evalscope.metrics.aggregators import BayesAtN
from evalscope.metrics.semantics import get_semantics_resolver
from evalscope.models.mockllm import MockLLM
from evalscope.report import Report
from evalscope.report.generator import ReportGenerator


def make_scores(rows: list[list[float]], metric: str = 'accuracy') -> list[SampleScore]:
    return [
        SampleScore(
            score=Score(value={metric: value}),
            sample_id=f'{question}-{position}',
            group_id=question,
            generation_index=position,
        )
        for question, row in enumerate(rows)
        for position, value in enumerate(row)
    ]


@pytest.mark.parametrize(
    'rows,expected_mean,expected_variance',
    [
        ([[0, 0], [1, 1]], Fraction(1, 2), Fraction(3, 160)),
        ([[0, 1], [0, 1]], Fraction(1, 2), Fraction(1, 40)),
        ([[0]], Fraction(1, 3), Fraction(1, 18)),
        ([[1]], Fraction(2, 3), Fraction(1, 18)),
        ([[0, 1, 1], [1, 1, 1]], Fraction(7, 10), Fraction(1, 60)),
    ],
)
def test_independent_beta_moments(rows: list[list[float]], expected_mean: Fraction, expected_variance: Fraction) -> None:
    mean, sd = get_aggregation('bayes_at_n')(repeats=len(rows[0]))(make_scores(rows))
    assert mean.score == pytest.approx(float(expected_mean), abs=1e-14)
    assert sd.score == pytest.approx(sqrt(float(expected_variance)), abs=1e-14)
    assert mean.aggregation == 'bayes_at_n'
    assert sd.aggregation == 'bayes_at_n_sd'
    assert mean.dimensions == {'n': len(rows[0])}
    assert mean.num == sd.num == len(rows)
    assert mean.metadata['generations'] == sum(map(len, rows))
    assert sd.metadata['posterior_sd'] == sd.score


@pytest.mark.parametrize('metric', ['acc', 'accuracy', 'em', 'exact_match'])
def test_correctness_aliases_and_semantics(metric: str) -> None:
    mean, sd = BayesAtN(2)(make_scores([[0, 1]], metric))
    resolver = get_semantics_resolver()
    mean_semantics = resolver.resolve('general_mcq', mean.identity).semantics
    sd_semantics = resolver.resolve('general_mcq', sd.identity).semantics
    assert mean_semantics.kind == MetricKind.QUALITY
    assert mean_semantics.direction == MetricDirection.HIGHER_IS_BETTER
    assert sd_semantics.kind == MetricKind.DIAGNOSTIC
    assert sd_semantics.direction == MetricDirection.NONE
    assert 'posterior SD' in sd_semantics.metric_name


def test_question_and_trial_order_do_not_change_moments() -> None:
    scores = make_scores([[0, 0, 1], [1, 1, 1], [0, 1, 0]])
    expected = BayesAtN(3)(scores)
    actual = BayesAtN(3)(list(reversed(scores)))
    assert [item.score for item in actual] == [item.score for item in expected]


def test_representative_workload_matches_exact_rational_moments() -> None:
    n = 64
    rows = [[1] * (question % 65) + [0] * (n - question % 65) for question in range(130)]
    mean, sd = BayesAtN(n)(make_scores(rows))
    means = [Fraction(sum(row) + 1, n + 2) for row in rows]
    variances = [Fraction((sum(row) + 1) * (n - sum(row) + 1), (n + 2) ** 2 * (n + 3)) for row in rows]
    assert mean.score == pytest.approx(float(sum(means) / len(rows)), abs=1e-14)
    assert sd.score == pytest.approx(sqrt(float(sum(variances) / len(rows) ** 2)), abs=1e-14)


@pytest.mark.parametrize('repeats', [0, -1, True, 2.5])
def test_invalid_planned_count(repeats: Any) -> None:
    with pytest.raises(ValueError, match='positive integer'):
        BayesAtN(repeats)


@pytest.mark.parametrize('value', [-1, 2, 0.5, float('nan'), float('inf')])
def test_nonbinary_or_nonfinite_outcomes_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match='binary correctness'):
        BayesAtN()(make_scores([[value]]))


@pytest.mark.parametrize('rows', [[], [[0, 1], [1]], [[0, 1, 1]]])
def test_empty_ragged_or_excess_trials_are_rejected(rows: list[list[float]]) -> None:
    with pytest.raises(ValueError, match='Bayes@N'):
        BayesAtN(2)(make_scores(rows))


@pytest.mark.parametrize('field', ['sample_id', 'group_id', 'generation_index'])
def test_missing_identity_is_rejected(field: str) -> None:
    scores = make_scores([[0, 1]])
    setattr(scores[0], field, None)
    with pytest.raises(ValueError, match='Bayes@N'):
        BayesAtN(2)(scores)


@pytest.mark.parametrize('field', ['sample_id', 'generation_index'])
def test_duplicate_identity_is_rejected(field: str) -> None:
    scores = make_scores([[0, 1]])
    setattr(scores[1], field, getattr(scores[0], field))
    with pytest.raises(ValueError, match='distinct|duplicate'):
        BayesAtN(2)(scores)


def test_a_generation_gap_is_rejected() -> None:
    scores = make_scores([[0, 1]])
    scores[1].generation_index = 2
    with pytest.raises(ValueError, match='exactly repeats'):
        BayesAtN(3)(scores)


@pytest.mark.parametrize('failure', ['missing_key', 'excluded', 'all_missing', 'different_keys'])
def test_unavailable_grades_are_rejected(failure: str) -> None:
    scores = make_scores([[0, 1]])
    if failure == 'excluded':
        scores[0].score.status = ScoreStatus.EXCLUDED
    elif failure == 'different_keys':
        scores[0].score.value = {'exact_match': 1}
    else:
        scores[0].score.value = {}
        if failure == 'all_missing':
            scores[1].score.value = {}
    with pytest.raises(ValueError, match='usable correctness|same correctness'):
        BayesAtN(2)(scores)


def test_other_metrics_and_duplicate_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match='only binary accuracy'):
        BayesAtN()(make_scores([[1]], 'judge_score'))
    scores = make_scores([[1]])
    scores[0].score.value['acc'] = 1
    with pytest.raises(ValueError, match='aliases'):
        BayesAtN()(scores)


def config_for(data: Path, output: Path, **kwargs: Any) -> TaskConfig:
    return TaskConfig(
        model='bayes-at-n-test',
        eval_type='mock_llm',
        datasets=['general_mcq'],
        dataset_args={'general_mcq': {'local_path': str(data), 'subset_list': ['tiny'], 'aggregation': 'bayes_at_n'}},
        repeats=2,
        eval_batch_size=1,
        generation_config={'temperature': 0.7},
        judge={'strategy': 'rule'},
        work_dir=str(output),
        no_timestamp=True,
        enable_progress_tracker=False,
        collect_perf=False,
        **kwargs,
    )


@pytest.fixture
def local_data(tmp_path: Path) -> Path:
    data = tmp_path / 'data'
    data.mkdir()
    rows = [
        {'question': 'First question', 'A': 'yes', 'B': 'no', 'answer': 'A'},
        {'question': 'Second question', 'A': 'yes', 'B': 'no', 'answer': 'B'},
    ]
    (data / 'tiny_val.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return data


def test_public_run_task_report_and_cache_resume(local_data: Path, tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(MockLLM, 'default_output', 'A')
    work = tmp_path / 'run'
    run_task(config_for(local_data, work))
    report_path = work / 'reports' / 'bayes-at-n-test' / 'general_mcq.json'
    report = Report.from_json(str(report_path))
    assert report.primary_metric_identity.aggregation == 'bayes_at_n'
    assert report.primary_metric_identity.dimensions == {'n': 2}
    assert report.num == 2
    assert report.score == 0.5
    sd = next(metric for metric in report.metrics if metric.identity.aggregation == 'bayes_at_n_sd')
    assert sd.score == round(sqrt(3 / 160), 4)
    assert sd.categories[0].subsets[0].metadata['posterior_sd'] == pytest.approx(sqrt(3 / 160), abs=1e-14)
    assert len(report.to_dataframe()) == 2
    original = report.to_dict()

    review_path = work / 'reviews' / 'bayes-at-n-test' / 'general_mcq_tiny.jsonl'
    review_rows = review_path.read_text().splitlines()
    assert len(review_rows) == 4
    review_path.write_text('\n'.join(reversed(review_rows + [review_rows[0]])) + '\n')
    monkeypatch.setattr(MockLLM, 'generate', lambda *args, **kwargs: pytest.fail('Cached predictions must be reused.'))
    run_task(config_for(local_data, work, use_cache=str(work)))
    resumed = Report.from_json(str(report_path))
    assert resumed.to_dict() == original


def test_missing_entire_question_cannot_shrink_loaded_cohort(local_data: Path, tmp_path: Path) -> None:
    adapter = get_benchmark('general_mcq', config_for(local_data, tmp_path / 'run'))
    adapter.load_dataset()
    scores = [
        SampleScore(score=Score(value={'accuracy': 1}), sample_id=sample.id, group_id=sample.group_id,
                    generation_index=sample.id % 2)
        for sample in adapter.test_dataset['tiny']
    ]
    with pytest.raises(ValueError, match='every planned sample ID'):
        adapter.aggregate_subset('tiny', scores[:2])
    with pytest.raises(ValueError, match='every planned sample ID'):
        adapter.aggregate_subset('tiny', [])
    scores[0].group_id = 999
    with pytest.raises(ValueError, match='does not match'):
        adapter.aggregate_subset('tiny', scores)


def test_multiple_subsets_fail_before_dataset_io(local_data: Path, tmp_path: Path, monkeypatch: Any) -> None:
    config = config_for(local_data, tmp_path / 'run')
    config.dataset_args['general_mcq']['subset_list'] = ['first', 'second']
    adapter = get_benchmark('general_mcq', config)
    monkeypatch.setattr(adapter, 'load', lambda: pytest.fail('Scope validation must precede loading.'))
    with pytest.raises(ValueError, match='one selected subset'):
        adapter.load_dataset()


def test_collection_configuration_is_rejected_before_io(tmp_path: Path, monkeypatch: Any) -> None:
    config = TaskConfig(model='unused', eval_type='mock_llm', datasets=['data_collection'],
                        dataset_args={'data_collection': {'aggregation': 'bayes_at_n', 'subset_list': ['default']}})
    adapter = get_benchmark('data_collection', config)
    monkeypatch.setattr(adapter, 'load', lambda: pytest.fail('Scope validation must precede loading.'))
    with pytest.raises(ValueError, match='dataset collections'):
        adapter.load_dataset()


def test_manual_report_cannot_average_subset_posterior_sd() -> None:
    adapter = SimpleNamespace(name='general_mcq')
    with pytest.raises(ValueError, match='posterior SD cannot be averaged'):
        ReportGenerator.generate_report(
            {'first': BayesAtN(2)(make_scores([[0, 0]])), 'second': BayesAtN(2)(make_scores([[0, 1], [1, 1]]))},
            'test-model', adapter,
        )


def test_public_all_error_run_cannot_disappear(local_data: Path, tmp_path: Path, monkeypatch: Any) -> None:
    def fail_generation(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('offline generation failure')

    monkeypatch.setattr(MockLLM, 'generate', fail_generation)
    with pytest.raises(ValueError, match='every planned sample ID'):
        run_task(config_for(local_data, tmp_path / 'failed-run', ignore_errors=True))


def test_collection_child_cannot_silently_ignore_bayes_at_n() -> None:
    from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
    from evalscope.constants import DataCollection

    config = TaskConfig(model='unused', eval_type='mock_llm', datasets=['data_collection'],
                        dataset_args={'general_mcq': {'aggregation': 'bayes_at_n', 'subset_list': ['tiny']}})
    adapter = get_benchmark('data_collection', config)
    adapter.test_dataset = DatasetDict({adapter.default_subset: MemoryDataset(samples=[
        Sample(input='question', metadata={DataCollection.INFO: {'dataset_name': 'general_mcq', 'subset_name': 'tiny'}})
    ])})
    with pytest.raises(ValueError, match='does not support dataset collections'):
        adapter._initialize_adapters()


def test_unsupported_metric_fails_before_dataset_io(local_data: Path, tmp_path: Path, monkeypatch: Any) -> None:
    config = config_for(local_data, tmp_path / 'run')
    config.dataset_args['general_mcq']['metric_list'] = ['bleu']
    adapter = get_benchmark('general_mcq', config)
    monkeypatch.setattr(adapter, 'load', lambda: pytest.fail('Metric validation must precede loading.'))
    with pytest.raises(ValueError, match='only binary accuracy'):
        adapter.load_dataset()


def test_partial_cache_resume_fills_one_missing_generation(local_data: Path, tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(MockLLM, 'default_output', 'A')
    work = tmp_path / 'partial-run'
    run_task(config_for(local_data, work))
    report_path = work / 'reports' / 'bayes-at-n-test' / 'general_mcq.json'
    original_metrics = Report.from_json(str(report_path)).metrics
    for kind in ['predictions', 'reviews']:
        cache = work / kind / 'bayes-at-n-test' / 'general_mcq_tiny.jsonl'
        records = [json.loads(line) for line in cache.read_text().splitlines()]
        cache.write_text(''.join(json.dumps(record) + '\n' for record in records if record['index'] != 1))

    calls = []
    original_generate = MockLLM.generate

    def count_generation(self: MockLLM, *args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        return original_generate(self, *args, **kwargs)

    monkeypatch.setattr(MockLLM, 'generate', count_generation)
    run_task(config_for(local_data, work, use_cache=str(work)))
    assert len(calls) == 1
    assert Report.from_json(str(report_path)).metrics == original_metrics
