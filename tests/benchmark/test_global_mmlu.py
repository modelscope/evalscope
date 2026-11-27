"""Global-MMLU: record mapping, subject-matched few-shot examples and CS/CA accuracy next to the overall accuracy."""

import pytest

from evalscope.api.benchmark import DataAdapter
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.global_mmlu.global_mmlu_adapter import GLOBAL_MMLU_LANGUAGES, GLOBAL_MMLU_LITE_LANGUAGES
from evalscope.config import TaskConfig
from evalscope.report.generator import ReportGenerator


def _adapter(name: str = 'global_mmlu', **dataset_args) -> DataAdapter:
    args = {name: dataset_args} if dataset_args else {}
    return get_benchmark(name, TaskConfig(model='mock', datasets=[name], dataset_args=args))


def _record(question: str, subject: str = 'astronomy', answer: str = 'C', label: str = '-') -> dict:
    return {
        'sample_id': f'{subject}/test/0',
        'subject': subject,
        'subject_category': 'STEM',
        'question': question,
        'option_a': 'a',
        'option_b': 'b',
        'option_c': 'c',
        'option_d': 'd',
        'answer': answer,
        'cultural_sensitivity_label': label,
    }


def _sample(adapter: DataAdapter, subset: str, record: dict) -> Sample:
    adapter.current_subset_name = subset
    return adapter.record_to_sample(record)


def test_language_tables() -> None:
    assert len(GLOBAL_MMLU_LANGUAGES) == 42 and len(GLOBAL_MMLU_LITE_LANGUAGES) == 23
    only_lite = set(GLOBAL_MMLU_LITE_LANGUAGES) - set(GLOBAL_MMLU_LANGUAGES)
    assert only_lite == {'cy', 'hu', 'my', 'or', 'sk', 'sq', 'tg'}
    # The cards rate some languages differently; each benchmark groups by its own card.
    assert _adapter().category_map['cs'] == 'high-resource'
    assert _adapter('global_mmlu_lite').category_map['cs'] == 'mid-resource'
    for name in ('global_mmlu', 'global_mmlu_lite'):
        adapter = _adapter(name)
        assert adapter.few_shot_num == 5 and adapter.train_split == 'dev' and adapter.eval_split == 'test'


def test_record_to_sample() -> None:
    sample = _sample(_adapter(), 'hi', _record(' प्रश्न? ', answer='D', label='CS'))
    assert sample.input == 'प्रश्न?'
    assert sample.choices == ['a', 'b', 'c', 'd']
    assert sample.target == 'D'
    assert sample.metadata['language'] == 'Hindi'
    assert sample.metadata['cultural_sensitivity_label'] == 'CS'
    assert sample.metadata['subject'] == 'astronomy'


@pytest.mark.parametrize('missing', [None, '', '\u00a0'])
def test_option_lost_in_translation_is_kept_empty(missing: str | None) -> None:
    # A few released rows (e.g. Amharic test row 739) have a None or empty option; the row stays, as in lm-eval.
    record = {**_record('ጥያቄ?', answer='B'), 'option_a': missing}
    sample = _sample(_adapter(), 'am', record)
    assert sample.choices == ['', 'b', 'c', 'd'] and sample.target == 'B'


def test_subject_comes_from_sample_id() -> None:
    # Global-MMLU-Lite's Hungarian rows name the subject in Hungarian; few-shot matching needs the English one.
    record = {**_record('kérdés?'), 'subject': 'csillagászat', 'sample_id': 'astronomy/test/58'}
    assert _sample(_adapter('global_mmlu_lite'), 'hu', record).metadata['subject'] == 'astronomy'


@pytest.mark.parametrize('question', [None, '', '  '])
def test_row_without_question_is_skipped(question) -> None:
    # Global-MMLU-Lite's Burmese test row 202 was released without its question.
    assert _sample(_adapter('global_mmlu_lite'), 'my', {**_record('x'), 'question': question}) == []


def _with_dev(adapter: DataAdapter, subset: str, dev: list) -> DataAdapter:
    adapter.fewshot_dataset = DatasetDict({subset: MemoryDataset([_sample(adapter, subset, r) for r in dev])})
    return adapter


DEV = [_record(f'algebra {i}', subject='abstract_algebra') for i in range(5)] + [
    _record(f'astronomy {i}', subject='astronomy') for i in range(5)
]


def test_few_shot_examples_come_from_the_same_subject() -> None:
    adapter = _with_dev(_adapter(), 'hi', DEV)
    prompt = adapter.process_sample_input(_sample(adapter, 'hi', _record('test question')), 'hi')
    assert all(f'astronomy {i}' in prompt for i in range(5))
    assert 'algebra' not in prompt
    assert prompt.rstrip().count('ANSWER: C') == 5  # one demonstrated answer per example

    adapter = _with_dev(_adapter(few_shot_num=2), 'hi', DEV)
    prompt = adapter.process_sample_input(_sample(adapter, 'hi', _record('test question')), 'hi')
    assert 'astronomy 1' in prompt and 'astronomy 2' not in prompt


def test_missing_subject_examples_raise() -> None:
    adapter = _with_dev(_adapter(), 'hi', DEV[:5])
    with pytest.raises(ValueError, match='has 0 `dev` questions for subject `astronomy`'):
        adapter.process_sample_input(_sample(adapter, 'hi', _record('test question')), 'hi')


def test_welsh_has_no_dev_split_and_runs_zero_shot() -> None:
    adapter = _adapter('global_mmlu_lite')
    assert len(adapter.load_fewshot_subset('cy', data_loader=None)) == 0
    adapter.fewshot_dataset = DatasetDict({'cy': MemoryDataset([])})
    sample = _sample(adapter, 'cy', _record('cwestiwn'))
    assert adapter.process_sample_input(sample, 'cy') == adapter.format_prompt_template(sample)


def _score(value: float, label: str) -> SampleScore:
    return SampleScore(
        score=Score(value={'acc': value}),
        sample_id=0,
        sample_metadata={'cultural_sensitivity_label': label},
    )


def test_cultural_sensitivity_accuracy() -> None:
    scores = [_score(1, 'CS'), _score(0, 'CS'), _score(1, 'CA'), _score(1, 'CA'), _score(0, 'CA'), _score(0, '-')]
    aggregates = {a.dimensions['cultural_sensitivity']: a for a in _adapter().aggregate_scores(scores)}
    assert set(aggregates) == {'all', 'cs', 'ca'}
    assert aggregates['all'].score == pytest.approx(3 / 6) and aggregates['all'].num == 6
    assert aggregates['cs'].score == pytest.approx(1 / 2) and aggregates['cs'].num == 2
    assert aggregates['ca'].score == pytest.approx(2 / 3) and aggregates['ca'].num == 3

    # Unannotated questions only count towards the overall accuracy.
    aggregates = _adapter().aggregate_scores([_score(1, '-')])
    assert [a.dimensions for a in aggregates] == [{'cultural_sensitivity': 'all'}]


def test_bayes_at_n_preserves_the_primary_overall_cohort() -> None:
    name = 'global_mmlu_lite'
    config = TaskConfig(
        model='mock',
        eval_type='mock_llm',
        datasets=[name],
        repeats=2,
        dataset_args={name: {'subset_list': ['en'], 'aggregation': 'bayes_at_n'}},
    )
    adapter = get_benchmark(name, config)
    scores = [
        SampleScore(
            score=Score(value={'acc': value}),
            sample_id=i,
            group_id=0,
            generation_index=i,
            sample_metadata={'cultural_sensitivity_label': 'CS'},
        )
        for i, value in enumerate([0, 1])
    ]
    report = ReportGenerator.generate_report({'en': adapter.aggregate_subset('en', scores)}, 'mock', adapter)
    assert report.primary_metric_identity.aggregation == 'bayes_at_n'
    assert report.primary_metric_identity.dimensions == {'cultural_sensitivity': 'all', 'n': 2}
    assert report.score == pytest.approx(0.5)
    assert report.num == 1
