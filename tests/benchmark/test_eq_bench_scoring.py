"""EQ-Bench v2 scoring, exclusions and report scale."""
from pathlib import Path
from typing import Any

import pytest

from evalscope import run_task
from evalscope.api.benchmark import DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus
from evalscope.metrics.semantics.formatting import format_metric_value
from evalscope.metrics.semantics.ranking import bounded_quality_ratio
from evalscope.report import Report

RECORD = {
    'prompt': 'Rate the emotions of Claudia at the end of the dialogue.',
    'reference_answer': "{'emotion1': 'Remorseful', 'emotion2': 'Indifferent', 'emotion3': 'Affectionate', "
    "'emotion4': 'Annoyed', 'emotion1_score': 2, 'emotion2_score': 3, 'emotion3_score': 0, 'emotion4_score': 5}",
    'reference_answer_fullscale': "{'emotion1': 'Remorseful', 'emotion2': 'Indifferent', 'emotion3': 'Affectionate', "
    "'emotion4': 'Annoyed', 'emotion1_score': 0, 'emotion2_score': '6', 'emotion3_score': 0, 'emotion4_score': '7'}",
}


@pytest.fixture
def adapter() -> DefaultDataAdapter:
    return get_benchmark('eq_bench', TaskConfig(model='mock', eval_type='mock_llm', datasets=['eq_bench']))


def evaluate(adapter: DefaultDataAdapter, completion: str, record: dict[str, Any] = RECORD) -> SampleScore:
    sample = adapter.record_to_sample(record)
    sample.id = 0
    state = TaskState(
        model='mock', sample=sample, output=ModelOutput.from_content(model='mock', content=completion), completed=True
    )
    return adapter.calculate_metrics(state)


def test_answer_far_from_reference_scores_below_zero(adapter: DefaultDataAdapter) -> None:
    # Remorseful and Annoyed are each 7 off: 10 - (7 + 7) * 0.7477 = -0.4678 on the 0-10 scale.
    completion = 'Remorseful: 7\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 0'

    assert evaluate(adapter, completion).score.main_value == pytest.approx((10 - 14 * 0.7477) / 10)


def test_exact_answer_scores_one(adapter: DefaultDataAdapter) -> None:
    result = evaluate(adapter, 'Remorseful: 0\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 7')
    assert result.score.main_value == pytest.approx(1.0)
    assert result.score.status is ScoreStatus.SUCCESS
    assert not result.score.metadata.get('metric_unavailable')


@pytest.mark.parametrize('completion', [
    '',
    'I cannot answer this question.',
    '[ERROR] request timed out',
    'Remorseful: 0\nIndifferent: 6\nAffectionate: 0',
    'Remorseful: 11\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 7',
])
def test_invalid_answer_is_excluded(adapter: DefaultDataAdapter, completion: str) -> None:
    result = evaluate(adapter, completion)
    assert result.score.status is ScoreStatus.EXCLUDED
    assert result.score.value == {}
    assert result.score.metadata['metric_unavailable']
    assert adapter.aggregate_scores([result]) == []


@pytest.mark.parametrize('reference', [None, '', '{}', 'not a reference', "{'emotion1': 'Remorseful'}"])
def test_missing_or_incomplete_v2_reference_does_not_fall_back_to_v1(
    adapter: DefaultDataAdapter, reference: Any
) -> None:
    # These ratings normalize to the v1 reference but must not enter a v2 average.
    record = {**RECORD, 'reference_answer_fullscale': reference}
    result = evaluate(adapter, 'Remorseful: 4\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 10', record)
    assert result.score.status is ScoreStatus.EXCLUDED
    assert result.score.value == {}
    assert result.score.metadata['metric_unavailable']


@pytest.mark.parametrize('failure', [None, ValueError('invalid reference')])
def test_scoring_failure_is_excluded(
    adapter: DefaultDataAdapter, monkeypatch: pytest.MonkeyPatch, failure: Any
) -> None:
    from evalscope.benchmarks.eq_bench import answer_validation

    def fail(*args: Any) -> None:
        if isinstance(failure, Exception):
            raise failure
        return None

    monkeypatch.setattr(answer_validation, 'calculate_score_fullscale', fail)
    result = evaluate(adapter, 'Remorseful: 0\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 7')
    assert result.score.status is ScoreStatus.EXCLUDED
    assert result.score.value == {}
    assert result.score.metadata['metric_unavailable']


@pytest.mark.parametrize('completion, expected, display', [
    ('Remorseful: 0\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 7', 1.0, '100%'),
    ('Remorseful: 7\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 0', -0.04678, '-4.7%'),
])
def test_report_averages_only_valid_scores_and_preserves_the_official_scale(
    adapter: DefaultDataAdapter, completion: str, expected: float, display: str
) -> None:
    scores = [evaluate(adapter, completion), evaluate(adapter, 'unparseable')]
    aggregate, = adapter.aggregate_scores(scores)
    assert aggregate.num == 1
    assert aggregate.score == pytest.approx(expected)

    report = adapter._on_generate_report({'default': [aggregate]}, model_name='mock')
    restored = Report.model_validate_json(report.model_dump_json())
    metric = restored.metrics[0]
    assert metric.score == round(expected, 4)
    assert format_metric_value(metric.score, metric.semantics) == display
    assert metric.semantics.value_range.min == -1.9908
    assert metric.semantics.value_range.max == 1.0
    if expected == 1.0:
        assert bounded_quality_ratio(metric.score, metric.semantics) == 1.0


def test_legacy_report_uses_the_same_storage_scale() -> None:
    report = Report.from_dict({
        'dataset_name': 'eq_bench',
        'metrics': [{'name': 'mean_eq_bench_score', 'score': 1.0, 'categories': []}],
    })
    metric = report.metrics[0]
    assert format_metric_value(metric.score, metric.semantics) == '100%'


def test_all_invalid_answers_produce_a_no_score_report(
    adapter: DefaultDataAdapter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def load_dataset(self: DefaultDataAdapter) -> DatasetDict:
        sample = self.record_to_sample(RECORD)
        sample.id = sample.group_id = 0
        return DatasetDict({'default': MemoryDataset([sample])})

    monkeypatch.setattr(type(adapter), 'load_dataset', load_dataset)
    reports = run_task(TaskConfig(
        model='eq-bench-invalid-output',
        eval_type='mock_llm',
        datasets=['eq_bench'],
        collect_perf=False,
        no_timestamp=True,
        work_dir=str(tmp_path),
    ))
    assert reports['eq_bench'].score is None
    assert reports['eq_bench'].metrics == []
    assert (tmp_path / 'reports' / 'eq-bench-invalid-output' / 'eq_bench.json').exists()
