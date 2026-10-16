"""Regression checks for native decision scoring under standard dataset transforms."""

import json
from pathlib import Path

import pytest

from evalscope import TaskConfig
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.model.decision import DecisionAnswer, DecisionInput
from evalscope.api.registry import get_benchmark
from evalscope.constants import ScoreStatus
from evalscope.metrics.decision import DecisionAggregator, decision_score
from evalscope.models.utils.decision import build_systemone_request, parse_decision_response


def test_native_decision_uses_standard_system_prompt_preparation(tmp_path: Path) -> None:
    record = {
        'state': 'Facts.', 'question': {'type': 'noul', 'instructions': 'Is it true?'}, 'expected': 'yes',
    }
    (tmp_path / 'default_test.jsonl').write_text(json.dumps(record) + '\n')
    adapter = get_benchmark('general_decision', TaskConfig(
        model='test', eval_type='systemone_api', datasets=['general_decision'],
        dataset_args={'general_decision': {'local_path': str(tmp_path), 'system_prompt': 'Use only the facts.'}},
    ))
    sample = adapter.load_dataset()['default'][0]
    assert [message.role for message in sample.input] == ['system', 'user']
    assert sample.input[0].text == 'Use only the facts.'
    assert sample.input[1].internal['systemone']['instructions'] == 'Is it true?'


@pytest.mark.parametrize('kind', ['choice', 'noul', 'score'])
def test_jevbench_accepts_lossless_mirror_records_and_original_json(kind: str) -> None:
    adapter = get_benchmark('jevbench', TaskConfig(model='test', eval_type='systemone_api', datasets=['jevbench']))
    question = {'type': kind, 'instructions': '按材料判断。'}
    if kind == 'choice':
        question['criteria'] = {'A': '第一项', 'B': '第二项'}
    elif kind == 'score':
        question['criteria'] = ['不符合', '部分符合', '完全符合']
    record = {
        'id': 'public-source', 'state': {'text': '公开材料'}, 'question': question,
        'expected': 'A' if kind == 'choice' else 'yes' if kind == 'noul' else '1',
        'family': 'source-family', 'tier': 'hard',
    }
    original = adapter.record_to_sample(record)
    mirrored = adapter.record_to_sample({'record_json': json.dumps(record, ensure_ascii=False)})
    exclude = {'input': {'__all__': {'id'}}}
    assert mirrored.model_dump(exclude=exclude) == original.model_dump(exclude=exclude)


@pytest.mark.parametrize(
    ('hub', 'args', 'expected'),
    [
        ('modelscope', {}, 'train'),
        ('huggingface', {}, 'test'),
        ('huggingface', {'eval_split': 'train'}, 'train'),
        ('huggingface', {'dataset_id': 'custom/decisions'}, 'train'),
    ],
)
def test_open_decisions_split_follows_source_and_explicit_configuration(hub: str, args: dict, expected: str) -> None:
    adapter = get_benchmark('open_decisions', TaskConfig(
        model='test', datasets=['open_decisions'], dataset_hub=hub,
        dataset_args={'open_decisions': args},
    ))
    assert adapter.eval_split == expected


def test_decision_mcq_preserves_an_explicit_primary_metric() -> None:
    adapter = get_benchmark('boolq', TaskConfig(
        model='test', eval_type='systemone_api', datasets=['boolq'],
        dataset_args={'boolq': {'metric_list': ['nll'], 'primary_metric': 'nll'}},
    ))
    assert adapter.primary_metric.name == 'nll'


@pytest.mark.parametrize('operation', ['statistics', 'example'])
def test_decision_documentation_loads_using_its_declared_api(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evalscope.utils.doc_utils.benchmark_stats import compute_benchmark_statistics, get_sample_example

    adapter = get_benchmark('stsb')
    sample = adapter.record_to_sample({'sentence1': 'One.', 'sentence2': 'Two.', 'score': 0.5})
    dataset = DatasetDict({'default': MemoryDataset([sample])})
    monkeypatch.setattr(adapter, 'load', lambda: (dataset, None))
    if operation == 'statistics':
        assert compute_benchmark_statistics(adapter).total_samples == 1
    else:
        assert get_sample_example(adapter) is not None


@pytest.mark.parametrize('name', ['jevbench', 'open_decisions', 'stsb', 'general_decision'])
def test_typed_benchmarks_reject_chat_before_loading(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = get_benchmark(name, TaskConfig(model='test', eval_type='openai_api', datasets=[name]))

    def forbidden_load() -> None:
        pytest.fail('An incompatible chat run must not download the dataset.')

    monkeypatch.setattr(adapter, 'load', forbidden_load)
    with pytest.raises(ValueError, match='systemone_api or autotrust_api'):
        adapter.load_dataset()


def test_boolq_noul_gold_follows_semantics_after_choice_shuffle() -> None:
    adapter = get_benchmark('boolq', TaskConfig(model='test', eval_type='systemone_api', datasets=['boolq'],
        dataset_args={'boolq': {'extra_params': {'decision_kind': 'noul'}}}))
    sample = adapter.record_to_sample({'passage': 'Paris is in France.', 'question': 'Is Paris in France?', 'answer': True})
    sample.choices = ['No', 'Yes']
    sample.target = 'B'
    answer = DecisionAnswer(type='noul', noul=0.9, probabilities={'no': 0.1, 'yes': 0.9})
    output = ModelOutput.from_content('test', 'ANSWER: yes')
    output.metadata = {'decision': answer.model_dump()}
    result = adapter.match_score('yes', 'yes', sample.target, TaskState(model='test', sample=sample, output=output))
    assert result.value['acc'] == 1


@pytest.mark.parametrize('metadata', [{}, {
    'choice_response': {
        'answers': {'answer': {'type': 'choice', 'choice': 'A', 'probabilities': {'A': 0.9, 'B': 0.1}}},
    },
}])
def test_mcq_missing_decision_is_excluded(metadata: dict) -> None:
    adapter = get_benchmark('boolq', TaskConfig(model='test', eval_type='systemone_api', datasets=['boolq']))
    sample = adapter.record_to_sample({'passage': 'Fact.', 'question': 'True?', 'answer': True})
    output = ModelOutput.from_content('test', 'ANSWER: A')
    output.metadata = metadata
    state = TaskState(model='test', sample=sample, output=output, completed=True)
    result = adapter.calculate_metrics(state)
    assert result.score.status is ScoreStatus.EXCLUDED
    assert result.score.metadata['metric_unavailable'] is True


def test_noul_conversion_requires_an_explicit_adapter_capability() -> None:
    config = TaskConfig(
        model='test',
        eval_type='systemone_api',
        datasets=['boolq'],
        dataset_args={'boolq': {'extra_params': {'decision_kind': 'noul'}}},
    )
    adapter = get_benchmark('boolq', config)
    adapter.decision_noul_positive_choice = None
    sample = Sample(input='Question?', choices=['Yes', 'No'], target='A')
    with pytest.raises(ValueError, match='no audited Noul conversion'):
        adapter.build_systemone_messages(sample, 'default')


def test_banking77_macro_f1_uses_intents_after_choice_shuffle() -> None:
    from evalscope.benchmarks.banking77.banking77_adapter import INTENT_LABELS
    from evalscope.utils.multi_choices import answer_character

    adapter = get_benchmark('banking77', TaskConfig(model='test', eval_type='systemone_api', datasets=['banking77']))
    scores = []
    # Both requests select position A, but they represent two distinct correctly predicted intents.
    for index in [0, 1]:
        sample = adapter.record_to_sample({'text': 'Support request', 'label': index})
        sample.choices[0], sample.choices[index] = sample.choices[index], sample.choices[0]
        sample.target = 'A'
        probabilities = {answer_character(i): float(i == 0) for i in range(77)}
        output = ModelOutput.from_content('test', 'ANSWER: A')
        output.metadata = {'decision': DecisionAnswer(choice='A', probabilities=probabilities).model_dump()}
        score = adapter.match_score('A', 'A', sample.target, TaskState(model='test', sample=sample, output=output))
        scores.append(SampleScore(sample_id=index, score=score))
    metrics = DecisionAggregator()(scores)
    assert next(m.score for m in metrics if m.metric_name == 'macro_f1') == pytest.approx(2 / len(INTENT_LABELS))


@pytest.mark.parametrize('name', ['jevbench', 'open_decisions'])
def test_repeated_official_tasks_keep_independent_predictions(name: str) -> None:
    adapter = get_benchmark(name, TaskConfig(model='test', eval_type='systemone_api', datasets=[name]))
    samples, scores = [], []
    for index in [0, 1]:
        sample = adapter.record_to_sample({'id': 'same-source', 'state': 'Fact.', 'family': 'unit', 'tier': 'hard',
            'question': {'type': 'choice', 'instructions': 'Select.', 'criteria': {'A': 'First', 'B': 'Second'}},
            'expected': 'A'})
        sample.id = index
        samples.append(sample)
        answer = DecisionAnswer(choice='A' if index == 0 else 'B', probabilities={'A': float(index == 0), 'B': float(index == 1)})
        scores.append(SampleScore(sample_id=index, sample_metadata=sample.metadata,
                                 score=decision_score(answer, 'A', ['A', 'B'], family='unit', tier='hard')))
    adapter.test_dataset = DatasetDict({'default': MemoryDataset(samples)})
    adapter.current_subset_name = 'loading-another-subset'
    metrics = adapter.aggregate_subset('default', scores)
    assert adapter.current_subset_name == 'loading-another-subset'
    if name == 'jevbench':
        assert next(m.score for m in metrics if m.metric_name == 'public_accuracy') == 0.5
    else:
        assert next(m.score for m in metrics if m.metric_name == 'intelligence') == 0
    assert all(m.num == 2 for m in metrics if m.aggregation == 'official')


def test_risk_coverage_does_not_depend_on_tied_sample_order() -> None:
    scores = [SampleScore(sample_id=index, score=decision_score(
        DecisionAnswer(choice='A', probabilities={'A': 0.9, 'B': 0.1}), gold, ['A', 'B']))
        for index, gold in enumerate(['A', 'B'])]
    first = next(m for m in DecisionAggregator()(scores) if m.metric_name == 'aurc')
    second = next(m for m in DecisionAggregator()(list(reversed(scores))) if m.metric_name == 'aurc')
    assert first.score == second.score == 0.5
    assert first.metadata['risk_coverage_top_probability'] == [{'coverage': 1.0, 'risk': 0.5}]


def test_jevbench_rounding_band_is_not_rejected_by_transport() -> None:
    request = DecisionInput(kind='choice', state='Fact.', instructions='Select.', criteria={'A': 'First', 'B': 'Second'})
    raw = {'model': 'test', 'answers': {'answer': {
        'type': 'choice', 'choice': 'A', 'probabilities': {'A': 0.67, 'B': 0.345}}}}
    answer = parse_decision_response(raw, request)
    adapter = get_benchmark('jevbench', TaskConfig(model='test', eval_type='systemone_api', datasets=['jevbench']))
    sample = adapter.record_to_sample({'id': 'rounded', 'state': 'Fact.',
        'question': build_systemone_request(request, 'test')['questions']['answer'], 'expected': 'A'})
    output = ModelOutput.from_content('test', 'ANSWER: A')
    output.metadata = {'decision': answer.model_dump()}
    score = adapter.match_score('A', 'A', 'A', TaskState(model='test', sample=sample, output=output))
    assert score.value['acc'] == 1
    assert score.metadata['probabilities_normalized_for_scoring']
    assert score.metadata['decision']['raw_probabilities'] == {'A': 0.67, 'B': 0.345}


def test_wide_options_do_not_justify_impossible_rounding() -> None:
    probabilities = {str(i): float(i < 2) for i in range(255)}
    with pytest.raises(ValueError, match='rounding tolerance'):
        DecisionAnswer(choice='0', probabilities=probabilities)


@pytest.mark.parametrize('value', [True, False, '0.9'])
def test_classification_labels_are_not_coerced_into_noul_probabilities(value: object) -> None:
    request = DecisionInput(kind='noul', state='Fact.', instructions='Is it true?')
    with pytest.raises(ValueError):
        parse_decision_response({'model': 'test', 'answers': {'answer': {'type': 'noul', 'noul': value}}}, request)


@pytest.mark.parametrize('value', [True, '0.9'])
def test_choice_probabilities_must_be_json_numbers(value: object) -> None:
    with pytest.raises(ValueError):
        DecisionAnswer(choice='A', probabilities={'A': value, 'B': 0.1})


@pytest.mark.parametrize('change', [
    {'probabilities': [True, False]},
    {'probabilities': ['1.0', 0.0]},
    {'probabilities': [float('nan'), 0.0]},
    {'choice_index': True},
    {'choice_index': '0'},
    {'choice_index': 2},
    {'choice': 'Second'},
    {'options': ['Second', 'First']},
])
def test_autotrust_response_contract_rejects_invalid_arrays_and_selections(change: dict) -> None:
    request = DecisionInput(state='Fact.', instructions='Choose.', criteria={'A': 'First', 'B': 'Second'})
    raw = {'model': 'test', 'kind': 'choice', 'options': ['First', 'Second'],
           'probabilities': [1.0, 0.0], 'choice_index': 0, 'choice': 'First', **change}
    with pytest.raises(ValueError):
        parse_decision_response(raw, request, 'autotrust')


def test_mcq_probability_order_follows_the_question_not_the_response() -> None:
    adapter = get_benchmark('boolq', TaskConfig(model='test', eval_type='systemone_api', datasets=['boolq']))
    sample = adapter.record_to_sample({'passage': 'Fact.', 'question': 'True?', 'answer': True})
    output = ModelOutput.from_content('test', 'ANSWER: A')
    output.metadata = {'decision': DecisionAnswer(choice='A', probabilities={'B': 0.1, 'A': 0.9}).model_dump()}
    score = adapter.match_score('A', 'A', sample.target, TaskState(model='test', sample=sample, output=output))
    assert score.metadata['decision']['labels'] == ['A', 'B']
    assert score.metadata['decision']['probabilities'] == [0.9, 0.1]
    assert list(score.metadata['decision']['raw_probabilities']) == ['B', 'A']


def test_open_decisions_uses_raw_probabilities_like_official_scorer() -> None:
    adapter = get_benchmark('open_decisions', TaskConfig(model='test', eval_type='systemone_api', datasets=['open_decisions']))
    sample = adapter.record_to_sample({'id': 'rounded-noul', 'state': 'Fact.', 'tier': 'hard',
        'question': {'type': 'noul', 'instructions': 'Is it true?'}, 'expected': 'no'})
    answer = DecisionAnswer(type='noul', noul=0.496, probabilities={'no': 0.496, 'yes': 0.496})
    output = ModelOutput.from_content('test', 'ANSWER: no')
    output.metadata = {'decision': answer.model_dump()}
    score = adapter.match_score('no', 'no', 'no', TaskState(model='test', sample=sample, output=output))
    assert score.value['acc'] == 1
    assert score.value['brier'] == pytest.approx(0.496**2)
    assert not score.metadata['probabilities_normalized_for_scoring']
