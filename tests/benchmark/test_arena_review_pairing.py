"""Arena imports must compare the same observation from each model's review cache."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from evalscope.api.evaluator import ReviewResult
from evalscope.api.messages import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentImage,
    ContentText,
    ContentVideo,
    PerformanceMetrics,
    messages_to_markdown,
)
from evalscope.api.metric import SampleScore, Score
from evalscope.api.metric.semantics import MetricIdentity
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.general_arena.general_arena_adapter import GeneralArenaAdapter
from evalscope.config import TaskConfig
from evalscope.metrics.judge.llm_judge import LLMJudge
from evalscope.metrics.semantics import get_semantics_resolver
from evalscope.report.report import Category, Metric, Report, Subset

PAIR_SUBSET = 'general_qa&example@candidate&baseline'


@pytest.fixture(autouse=True)
def forbid_judge_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_judge_call(*args: Any, **kwargs: Any) -> None:
        pytest.fail('Importing arena reviews must not call a judge')

    monkeypatch.setattr(LLMJudge, 'generate', unexpected_judge_call)


def _review(index: int, answer: str, question: str | None = None, *, modern: bool = False) -> dict[str, Any]:
    messages = [ChatMessageUser(content=question or f'Question {index}', source='input')]
    if modern:
        messages.append(ChatMessageAssistant(content=answer, source='generate'))
    return ReviewResult(
        index=index,
        messages=messages,
        sample_score=SampleScore(
            sample_id=index,
            score=Score(prediction=answer, extracted_prediction=answer),
        ),
    ).model_dump(mode='json')


def _reviews(indices: list[int], model: str) -> list[dict[str, Any]]:
    return [_review(index, f'{model} answer {index}') for index in indices]


def _adapter(
    tmp_path: Path,
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
) -> GeneralArenaAdapter:
    pytest.importorskip('sklearn')
    models = []
    identity = MetricIdentity(name='accuracy', aggregation='mean')
    for name, rows in [('candidate', candidate), ('baseline', baseline)]:
        report_path = tmp_path / 'reports' / name
        review_path = tmp_path / 'reviews' / name / 'general_qa_example.jsonl'
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        Report(
            dataset_name='general_qa',
            model_name=name,
            metrics=[
                Metric(
                    identity=identity,
                    semantics=get_semantics_resolver().resolve('general_qa', identity).semantics,
                    categories=[Category(name=('default',), subsets=[Subset(name='example', num=len(rows))])],
                )
            ],
        ).to_json(str(report_path / 'general_qa.json'))
        models.append({'name': name, 'report_path': str(report_path)})

    config = TaskConfig(
        model='arena',
        eval_type='mock_llm',
        datasets=['general_arena'],
        dataset_args={'general_arena': {'extra_params': {'models': models, 'baseline': 'baseline'}}},
    )
    return get_benchmark('general_arena', config)


def _pairs(adapter: GeneralArenaAdapter) -> list[dict[str, Any]]:
    adapter._check_names()
    adapter._check_reports()
    adapter._check_datasets()
    return adapter._build_pair_wise_data(adapter._load_common_datasets())[PAIR_SUBSET]


@pytest.mark.parametrize('candidate_order,baseline_order', [([0, 1, 2], [0, 1, 2]), ([2, 0, 1], [1, 2, 0])])
def test_complete_reviews_pair_by_observation(
    tmp_path: Path, candidate_order: list[int], baseline_order: list[int]
) -> None:
    adapter = _adapter(tmp_path, _reviews(candidate_order, 'candidate'), _reviews(baseline_order, 'baseline'))
    pairs = _pairs(adapter)
    assert [(pair['answer_1'], pair['answer_2']) for pair in pairs] == [
        (f'candidate answer {index}', f'baseline answer {index}') for index in range(3)
    ]
    assert all(pair['model_1'] == 'candidate' and pair['model_2'] == 'baseline' for pair in pairs)


@pytest.mark.parametrize(
    'candidate_indices,baseline_indices',
    [
        ([0, 2], [0, 1, 2]),
        ([0, 1, 2], [0, 2]),
        ([0, 1], [0, 1, 2]),
        ([0, 1, 2], [0, 1]),
        ([0, 2], [0, 1]),
        ([0, 1], [0, 2]),
        ([], [0, 1]),
        ([0, 1], []),
        ([], []),
    ],
)
def test_incomplete_or_incompatible_review_coverage_is_rejected(
    tmp_path: Path, candidate_indices: list[int], baseline_indices: list[int]
) -> None:
    adapter = _adapter(tmp_path, _reviews(candidate_indices, 'candidate'), _reviews(baseline_indices, 'baseline'))
    with pytest.raises(ValueError, match=r'(?i)missing|indices|index|coverage|empty|no review'):
        _pairs(adapter)


def test_same_index_with_different_input_is_rejected(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path,
        [_review(0, 'candidate answer', 'How do I grow tomatoes?')],
        [_review(0, 'baseline answer', 'How do I learn French?')],
    )
    with pytest.raises(ValueError, match=r'(?i)input|prompt|question'):
        _pairs(adapter)


def test_modern_generated_messages_do_not_make_matching_inputs_different(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path,
        [_review(0, 'Candidate response', modern=True)],
        [_review(0, 'Baseline response', modern=True)],
    )
    pairs = _pairs(adapter)
    assert len(pairs) == 1
    assert pairs[0]['answer_1'] == 'Candidate response'
    assert pairs[0]['answer_2'] == 'Baseline response'


def test_legacy_input_strings_and_missing_optional_identity_are_accepted(tmp_path: Path) -> None:
    candidate = _review(0, 'Candidate response')
    baseline = _review(0, 'Baseline response')
    for record in [candidate, baseline]:
        record.pop('messages')
        record['input'] = 'Legacy input'
        for field in ['sample_id', 'group_id', 'generation_index']:
            record['sample_score'].pop(field)
    pairs = _pairs(_adapter(tmp_path, [candidate], [baseline]))
    assert len(pairs) == 1
    assert 'Legacy input' in pairs[0]['question']
    assert (pairs[0]['answer_1'], pairs[0]['answer_2']) == ('Candidate response', 'Baseline response')


@pytest.mark.parametrize('legacy_side', ['candidate', 'baseline'])
def test_legacy_rendered_input_matches_modern_message_prefix(tmp_path: Path, legacy_side: str) -> None:
    prefix = [
        ChatMessageUser(content='Example question', source='input'),
        ChatMessageAssistant(content='Example answer', source='input'),
        ChatMessageUser(content='Current question', source='input'),
    ]
    records = {}
    for name in ['candidate', 'baseline']:
        record = _review(0, f'{name} response', modern=True)
        record['messages'] = [message.model_dump(mode='json') for message in prefix] + record['messages'][1:]
        if name == legacy_side:
            record.pop('messages')
            record['input'] = messages_to_markdown(prefix)
        records[name] = record
    pairs = _pairs(_adapter(tmp_path, [records['candidate']], [records['baseline']]))
    assert len(pairs) == 1
    assert (pairs[0]['answer_1'], pairs[0]['answer_2']) == ('candidate response', 'baseline response')


def test_unmarked_assistant_demonstrations_are_part_of_the_input(tmp_path: Path) -> None:
    candidate = _review(0, 'Candidate response', modern=True)
    baseline = _review(0, 'Baseline response', modern=True)
    for record, demonstration in [(candidate, 'One example answer'), (baseline, 'Different example answer')]:
        record['messages'].insert(
            0, ChatMessageAssistant(content=demonstration).model_dump(mode='json')
        )
    with pytest.raises(ValueError, match=r'(?i)input|prompt|question'):
        _pairs(_adapter(tmp_path, [candidate], [baseline]))


@pytest.mark.parametrize('collision', ['role', 'message-boundary'])
def test_matching_rendered_text_does_not_hide_different_message_structure(tmp_path: Path, collision: str) -> None:
    candidate, baseline = _review(0, 'Candidate response'), _review(0, 'Baseline response')
    if collision == 'role':
        candidate_messages = [ChatMessageUser(content='Follow this instruction')]
        baseline_messages = [ChatMessageSystem(content='Follow this instruction')]
    else:
        candidate_messages = [ChatMessageUser(content='First part\n\nSecond part')]
        baseline_messages = [ChatMessageUser(content='First part'), ChatMessageUser(content='Second part')]
    assert messages_to_markdown(candidate_messages) == messages_to_markdown(baseline_messages)
    candidate['messages'] = [message.model_dump(mode='json') for message in candidate_messages]
    baseline['messages'] = [message.model_dump(mode='json') for message in baseline_messages]
    with pytest.raises(ValueError, match=r'(?i)input|prompt|question'):
        _pairs(_adapter(tmp_path, [candidate], [baseline]))


@pytest.mark.parametrize('media', ['image-detail', 'video-segment'])
def test_matching_media_urls_do_not_hide_different_input_parameters(tmp_path: Path, media: str) -> None:
    candidate, baseline = _review(0, 'Candidate response'), _review(0, 'Baseline response')
    if media == 'image-detail':
        candidate_content = ContentImage(image='https://example.invalid/image.png', detail='low')
        baseline_content = ContentImage(image='https://example.invalid/image.png', detail='high')
    else:
        candidate_content = ContentVideo(video='https://example.invalid/video.mp4', format='mp4', start=0, end=5)
        baseline_content = ContentVideo(video='https://example.invalid/video.mp4', format='mp4', start=5, end=10)
    candidate_messages = [ChatMessageUser(content=[candidate_content])]
    baseline_messages = [ChatMessageUser(content=[baseline_content])]
    assert messages_to_markdown(candidate_messages) == messages_to_markdown(baseline_messages)
    candidate['messages'] = [message.model_dump(mode='json') for message in candidate_messages]
    baseline['messages'] = [message.model_dump(mode='json') for message in baseline_messages]
    with pytest.raises(ValueError, match=r'(?i)input|prompt|question'):
        _pairs(_adapter(tmp_path, [candidate], [baseline]))


def test_matching_structured_inputs_ignore_message_ids_and_run_metadata(tmp_path: Path) -> None:
    records = {}
    for index, name in enumerate(['candidate', 'baseline']):
        record = _review(0, f'{name} response', modern=True)
        prefix = [
            ChatMessageUser(
                content=[ContentText(text='Describe this'), ContentImage(image='https://example.invalid/image.png')],
                source='input' if name == 'candidate' else None,
                metadata={'run': name},
            ),
            ChatMessageAssistant(
                content='An example description',
                source='input' if name == 'candidate' else None,
                model=name,
                perf_metrics=PerformanceMetrics(latency=index + 1.0),
            ),
        ]
        record['messages'] = [message.model_dump(mode='json') for message in prefix] + record['messages'][1:]
        records[name] = record
    assert records['candidate']['messages'][0]['id'] != records['baseline']['messages'][0]['id']
    pairs = _pairs(_adapter(tmp_path, [records['candidate']], [records['baseline']]))
    assert len(pairs) == 1
    assert (pairs[0]['answer_1'], pairs[0]['answer_2']) == ('candidate response', 'baseline response')


@pytest.mark.parametrize('latest_is_modern', [True, False])
def test_duplicate_replacement_uses_latest_rows_legacy_status(tmp_path: Path, latest_is_modern: bool) -> None:
    legacy = _review(0, 'Legacy response')
    legacy.pop('messages')
    legacy['input'] = 'Same rendered input'
    modern = _review(0, 'Modern response')
    modern['messages'] = [ChatMessageSystem(content='Same rendered input').model_dump(mode='json')]
    baseline = _review(0, 'Baseline response', question='Same rendered input')
    candidate_rows = [legacy, modern] if latest_is_modern else [modern, legacy]
    adapter = _adapter(tmp_path, candidate_rows, [baseline])
    if latest_is_modern:
        with pytest.raises(ValueError, match=r'(?i)input|prompt|question'):
            _pairs(adapter)
    else:
        pairs = _pairs(adapter)
        assert len(pairs) == 1
        assert pairs[0]['answer_1'] == 'Legacy response'


def test_repeated_observations_with_distinct_indices_remain_distinct(tmp_path: Path) -> None:
    records = {}
    for name in ['candidate', 'baseline']:
        records[name] = [_review(index, f'{name} repeat {index}', 'Repeated question') for index in range(2)]
        for index, record in enumerate(records[name]):
            record['sample_score'].update(group_id=0, generation_index=index)
    pairs = _pairs(_adapter(tmp_path, records['candidate'], list(reversed(records['baseline']))))
    assert [(pair['answer_1'], pair['answer_2']) for pair in pairs] == [
        ('candidate repeat 0', 'baseline repeat 0'),
        ('candidate repeat 1', 'baseline repeat 1'),
    ]


@pytest.mark.parametrize('field', ['group_id', 'generation_index'])
def test_conflicting_supplied_repeat_identity_is_rejected(tmp_path: Path, field: str) -> None:
    candidate, baseline = _review(0, 'Candidate response'), _review(0, 'Baseline response')
    candidate['sample_score'][field] = 0
    baseline['sample_score'][field] = 1
    with pytest.raises(ValueError, match=r'(?i)group|generation|identity'):
        _pairs(_adapter(tmp_path, [candidate], [baseline]))


@pytest.mark.parametrize('field', ['sample_id', 'group_id', 'generation_index'])
@pytest.mark.parametrize('legacy_side', ['candidate', 'baseline'])
def test_optional_identity_missing_on_one_side_is_accepted(tmp_path: Path, field: str, legacy_side: str) -> None:
    records = {'candidate': _review(0, 'Candidate response'), 'baseline': _review(0, 'Baseline response')}
    for record in records.values():
        record['sample_score'][field] = 0
    records[legacy_side]['sample_score'].pop(field)
    pairs = _pairs(_adapter(tmp_path, [records['candidate']], [records['baseline']]))
    assert len(pairs) == 1


def test_numeric_string_sample_and_group_identities_match_integer_identities(tmp_path: Path) -> None:
    candidate, baseline = _review(1, 'Candidate response'), _review(1, 'Baseline response')
    candidate['sample_score'].update(sample_id='1', group_id='0')
    baseline['sample_score'].update(sample_id=1, group_id=0)
    assert len(_pairs(_adapter(tmp_path, [candidate], [baseline]))) == 1


@pytest.mark.parametrize('side', ['candidate', 'baseline'])
def test_conflicting_native_index_and_sample_id_is_rejected(tmp_path: Path, side: str) -> None:
    records = {'candidate': _review(0, 'Candidate response'), 'baseline': _review(0, 'Baseline response')}
    records[side]['sample_score']['sample_id'] = 9
    with pytest.raises(ValueError, match=r'(?i)sample|index|identity'):
        _pairs(_adapter(tmp_path, [records['candidate']], [records['baseline']]))


@pytest.mark.parametrize('side', ['candidate', 'baseline'])
def test_resumed_review_cache_uses_last_record_with_warning(
    tmp_path: Path, side: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = {'candidate': _reviews([0, 1], 'candidate'), 'baseline': _reviews([0, 1], 'baseline')}
    records[side].append(_review(0, f'{side} refreshed answer'))
    adapter = _adapter(tmp_path, records['candidate'], records['baseline'])
    monkeypatch.setattr(logging.getLogger('evalscope'), 'propagate', True)
    with caplog.at_level(logging.WARNING, logger='evalscope'):
        pairs = _pairs(adapter)
    assert len(pairs) == 2
    answer_field = 'answer_1' if side == 'candidate' else 'answer_2'
    assert pairs[0][answer_field] == f'{side} refreshed answer'
    assert pairs[1][answer_field] == f'{side} answer 1'
    assert any('duplicate' in record.message.lower() for record in caplog.records)


def test_load_converts_native_reports_and_reviews_to_arena_samples(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, _reviews([2, 0, 1], 'candidate'), _reviews([1, 2, 0], 'baseline'))
    dataset, train = adapter.load()
    assert train is None
    assert list(dataset.keys()) == [PAIR_SUBSET]
    assert len(dataset[PAIR_SUBSET]) == 3
    for index, sample in enumerate(dataset[PAIR_SUBSET]):
        assert f'Question {index}' in sample.input[0].content
        assert sample.target == f'baseline answer {index}'
        assert sample.metadata == {
            'answer_1': f'candidate answer {index}',
            'model_1': 'candidate',
            'model_2': 'baseline',
        }
