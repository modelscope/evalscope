"""Grouped metrics must keep independent generations separate across source rows."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from evalscope.api.benchmark import DataAdapter
from evalscope.api.dataset.builder import build_dataset_from_records
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

Case = tuple[str, str, list[dict[str, Any]], str, str]


@pytest.fixture(params=['visfactor', 'hallusion_bench', 'acebench'])
def grouped_case(request: pytest.FixtureRequest) -> Case:
    name = request.param
    if name == 'visfactor':
        records = [
            dict(index=i, category_id='CF1', category_name='Hidden Figures', eval_index=0,
                 question=f'Variant {i}', image=[], answer='T')
            for i in range(2)
        ]
        return name, 'default', records, '{"answer": true}', '{"answer": false}'
    if name == 'hallusion_bench':
        records = [
            dict(question=f'Question {i}', gt_answer='1', category='VD', subcategory='chart',
                 set_id='0', figure_id='1', question_id=str(i))
            for i in range(2)
        ]
        return name, 'default', records, 'YES', 'NO'
    category = 'normal_multi_turn_user_adjust'
    records = [
        dict(id=f'{category}_0_{i}', sub_category=f'data_{category}',
             question=f'user: Book in Athens. Step {i}.',
             function=[dict(name='book', description='Book a room.', parameters=dict(
                 type='object', properties={'city': {'type': 'string'}}, required=['city']))],
             rubric={'ground_truth': {'book': {'city': 'Athens'}}})
        for i in range(2)
    ]
    return name, category, records, "[book(city='Athens')]", "[book(city='Rome')]"


def _score_case(
    case: Case, repeats: int, predictions: list[str], concurrent: bool = False
) -> tuple[DataAdapter, list[SampleScore]]:
    name, subset, records, _, _ = case
    adapter = get_benchmark(name, TaskConfig(
        model='offline-scripted', datasets=[name], repeats=repeats,
        dataset_args={name: {'subset_list': [subset]}},
    ))
    dataset = build_dataset_from_records(
        records, adapter.record_to_sample, name=subset, location=None,
        limit=None, repeats=repeats, shuffle=False, seed=42,
    )
    states = [TaskState(
        model='offline-scripted', sample=sample, completed=True,
        output=ModelOutput.from_content('offline-scripted', prediction),
    ) for sample, prediction in zip(dataset, predictions)]
    assert len(states) == len(dataset) == len(predictions)
    if concurrent:
        with ThreadPoolExecutor(max_workers=4) as pool:
            scores = list(pool.map(adapter.calculate_metrics, states))
    else:
        scores = [adapter.calculate_metrics(state) for state in states]
    return adapter, scores


def _metrics(adapter: DataAdapter, subset: str, scores: list[SampleScore]) -> dict[str, Any]:
    aggregates = adapter.aggregate_subset(subset, scores)
    report = adapter.generate_report({subset: aggregates}, model_name='offline-scripted', output_dir='')
    result = {'report': report.score}
    for agg in aggregates:
        if adapter.name == 'hallusion_bench':
            if agg.dimensions.get('level') != 'overall':
                continue
            key = agg.dimensions['target']
        elif adapter.name == 'visfactor':
            if agg.aggregation != 'macro_mean':
                continue
            key = 'accuracy'
        else:
            key = agg.metric_name
        result[key] = (agg.score, agg.num)
    return result


@pytest.mark.parametrize('mode', ['serial', 'cached_reordered', 'concurrent'])
def test_grouped_repeats_match_independent_runs(grouped_case: Case, mode: str) -> None:
    name, subset, _, correct, wrong = grouped_case
    adapter, scores = _score_case(grouped_case, 2, [correct, wrong, correct, correct], mode == 'concurrent')
    assert [s.generation_index for s in scores] == [0, 1, 0, 1]
    assert [s.group_id for s in scores] == [0, 0, 1, 1]
    if mode != 'serial':
        scores = [SampleScore.model_validate_json(scores[i].model_dump_json()) for i in [3, 1, 0, 2]]
    actual = _metrics(adapter, subset, scores)
    independent = []
    for predictions in ([correct, correct], [wrong, correct]):
        single_adapter, single_scores = _score_case(grouped_case, 1, predictions)
        independent.append(_metrics(single_adapter, subset, single_scores))
    assert actual['report'] == pytest.approx(sum(m['report'] for m in independent) / 2)
    for key, metric in actual.items():
        if key == 'report':
            continue
        value, count = metric
        assert value == pytest.approx(sum(m[key][0] for m in independent) / 2)
        assert count == sum(m[key][1] for m in independent)
    if name == 'hallusion_bench':
        assert actual['figure'] == (0.5, 2)
        assert actual['question'] == (0.75, 4)
        assert actual['answer'] == (0.75, 4)
    else:
        assert actual['accuracy'] == (0.5, 2)
        if name == 'acebench':
            assert actual['process_acc'] == (0.75, 2)


def test_all_correct_repeats_count_group_attempts(grouped_case: Case) -> None:
    _, subset, _, correct, _ = grouped_case
    adapter, scores = _score_case(grouped_case, 2, [correct] * 4)
    actual = _metrics(adapter, subset, scores)
    assert actual['report'] == 1.0
    key = 'figure' if adapter.name == 'hallusion_bench' else 'accuracy'
    assert actual[key] == (1.0, 2)


def test_missing_attempt_does_not_shift_later_generations(grouped_case: Case) -> None:
    _, subset, _, correct, wrong = grouped_case
    adapter, scores = _score_case(grouped_case, 2, [correct, correct, wrong, correct])
    # Drop row 0 / generation 0. Its surviving generation 1 must not join row 1 / generation 0.
    scores = [SampleScore.model_validate_json(scores[i].model_dump_json()) for i in [3, 1, 2]]
    actual = _metrics(adapter, subset, scores)
    key = 'figure' if adapter.name == 'hallusion_bench' else 'accuracy'
    assert actual[key] == (0.5, 2)


@pytest.mark.parametrize('repeats', [1, 2])
def test_legacy_reviews_require_repeat_identity_only_for_repeats(grouped_case: Case, repeats: int) -> None:
    _, subset, _, correct, _ = grouped_case
    adapter, scores = _score_case(grouped_case, repeats, [correct] * (2 * repeats))
    # An old review can also be mixed with newly scored records during resume.
    legacy = scores[0].model_dump()
    legacy.pop('generation_index')
    scores[0] = SampleScore.model_validate(legacy)
    if repeats == 1:
        assert _metrics(adapter, subset, scores)['report'] == 1.0
    else:
        with pytest.raises(ValueError, match='generation_index.*rerun_review'):
            adapter.aggregate_scores(scores)


def test_visfactor_preserves_category_macro_with_unequal_coverage() -> None:
    adapter = get_benchmark('visfactor', TaskConfig(datasets=['visfactor'], repeats=2))
    scores = [SampleScore(
        score=Score(value={'accuracy': value}), generation_index=generation,
        sample_metadata={'category_id': category, 'category_name': category, 'eval_index': item},
    ) for category, item, generation, value in [
        ('CF1', 0, 0, 1), ('CF1', 1, 0, 0), ('CF1', 0, 1, 0), ('CS1', 0, 0, 1),
    ]]
    result = _metrics(adapter, 'default', scores)
    assert result['accuracy'] == (pytest.approx((1 / 3 + 1) / 2), 4)


def test_acebench_preserves_dialogue_weighted_process_accuracy() -> None:
    adapter = get_benchmark('acebench', TaskConfig(datasets=['acebench'], repeats=2))
    scores = [SampleScore(
        sample_id=index, score=Score(value={'acc': value}), generation_index=generation,
        sample_metadata={'test_category': 'normal_multi_turn_user_adjust', 'dialogue_id': dialogue},
    ) for index, (dialogue, generation, value) in enumerate([
        ('long', 0, 1), ('long', 0, 1), ('long', 1, 1), ('long', 1, 0),
        ('short', 0, 0), ('short', 1, 1),
    ])]
    result = _metrics(adapter, 'normal_multi_turn_user_adjust', scores)
    assert result['accuracy'] == (0.5, 4)
    assert result['process_acc'] == (0.625, 4)


def test_hallusion_repeats_preserve_category_and_no_figure_rules() -> None:
    adapter = get_benchmark('hallusion_bench', TaskConfig(datasets=['hallusion_bench'], repeats=2))
    scores = [SampleScore(
        score=Score(value={'acc': value}), generation_index=generation,
        sample_metadata={'category': category, 'subcategory': 'chart', 'set_id': '0',
                         'figure_id': figure, 'question_id': '0'},
    ) for category, figure, generation, value in [
        ('VD', '1', 0, 1), ('VD', '1', 1, 0),
        ('VS', '1', 0, 1), ('VS', '1', 1, 1),
        ('VS', '0', 0, 0), ('VS', '0', 1, 0),
    ]]
    result = _metrics(adapter, 'default', scores)
    assert result['figure'] == (0.75, 4)
    assert result['question'] == (0.25, 4)
    assert result['answer'] == (0.5, 6)
