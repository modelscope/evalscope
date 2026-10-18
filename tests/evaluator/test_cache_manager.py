"""Unit tests for cache resume behavior."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.evaluator.cache import CacheManager, ModelResult, ReviewResult
from evalscope.api.model import ModelOutput
from evalscope.utils.io_utils import OutputsStructure


def _make_task_state(sample_id: int, group_id: Optional[int] = None) -> TaskState:
    sample = Sample(id=sample_id, group_id=group_id, input=f'question-{sample_id}', target='answer')
    return TaskState(model='mock-model', sample=sample, completed=True)


def _review_row(sample_id: int, value: float) -> Dict[str, Any]:
    return {
        'index': sample_id,
        'target': 'answer',
        'messages': [],
        'sample_score': {
            'sample_id': sample_id,
            'score': {
                'value': {
                    'acc': value
                },
            },
        },
    }


def test_review_result_migrates_legacy_target_to_alternatives() -> None:
    legacy = ReviewResult.from_cache_item(_review_row(0, 1.0))
    current = ReviewResult.from_cache_item({**_review_row(1, 1.0), 'target': ['A', 'B']})

    assert legacy.target == ['answer']
    assert current.target == ['A', 'B']
    with pytest.raises(ValueError, match='must not be an empty list'):
        ReviewResult.from_cache_item({**_review_row(2, 1.0), 'target': []})


def _make_manager(tmp_path: Path) -> CacheManager:
    outputs = OutputsStructure(outputs_dir=str(tmp_path), is_make=True)
    return CacheManager(outputs=outputs, model_name='mock-model', benchmark_name='mock-bench')


def _write_review_cache(manager: CacheManager, subset: str, rows: list[Dict[str, Any]]) -> str:
    review_file = manager.get_review_cache_path(subset)
    with open(review_file, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    return review_file


@pytest.fixture
def caplog_evalscope(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> pytest.LogCaptureFixture:
    """Capture records from EvalScope's non-propagating logger."""
    monkeypatch.setattr(logging.getLogger('evalscope'), 'propagate', True)
    caplog.set_level(logging.WARNING, logger='evalscope')
    return caplog


def test_filter_review_cache_intersects_and_dedupes(
    tmp_path: Path,
    caplog_evalscope: pytest.LogCaptureFixture,
) -> None:
    manager = _make_manager(tmp_path)
    task_states = [_make_task_state(0), _make_task_state(1)]
    _write_review_cache(
        manager,
        'default',
        [
            _review_row(0, 1.0),
            _review_row(1, 0.0),
            _review_row(1, 1.0),
            _review_row(99, 0.5),
        ],
    )

    cached_scores, remaining_states = manager.filter_review_cache('default', task_states)

    assert [score.sample_id for score in cached_scores] == [0, 1]
    assert cached_scores[1].score.main_value == 1.0
    assert remaining_states == []
    assert 'Dropped 1 orphan and 1 duplicate rows' in caplog_evalscope.text


def test_filter_review_cache_does_not_merge_distinct_repeat_samples(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    task_states = [_make_task_state(10, group_id=5), _make_task_state(11, group_id=5)]
    _write_review_cache(manager, 'default', [_review_row(10, 1.0), _review_row(11, 0.0)])

    cached_scores, remaining_states = manager.filter_review_cache('default', task_states)

    assert [score.sample_id for score in cached_scores] == [10, 11]
    assert remaining_states == []


def test_filter_review_cache_returns_all_states_when_no_cache(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    task_states = [_make_task_state(0), _make_task_state(1)]

    cached_scores, remaining_states = manager.filter_review_cache('default', task_states)

    assert cached_scores == []
    assert remaining_states == task_states


def test_filter_review_cache_keeps_uncovered_states(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    task_states = [_make_task_state(0), _make_task_state(1), _make_task_state(2)]
    _write_review_cache(manager, 'default', [_review_row(2, 0.0)])

    cached_scores, remaining_states = manager.filter_review_cache('default', task_states)

    assert [score.sample_id for score in cached_scores] == [2]
    assert [state.sample_id for state in remaining_states] == [0, 1]


def test_filter_review_cache_skips_malformed_and_schema_invalid_rows(
    tmp_path: Path,
    caplog_evalscope: pytest.LogCaptureFixture,
) -> None:
    manager = _make_manager(tmp_path)
    task_states = [_make_task_state(0), _make_task_state(1)]
    review_file = _write_review_cache(
        manager,
        'default',
        [
            _review_row(0, 1.0),
            {'legacy_field': 'not a valid ReviewResult'},
            _review_row(1, 0.0),
        ],
    )
    with open(review_file, 'a', encoding='utf-8') as f:
        f.write('{"sample_score":')

    cached_scores, remaining_states = manager.filter_review_cache('default', task_states)

    assert [score.sample_id for score in cached_scores] == [0, 1]
    assert remaining_states == []
    assert 'Skipping invalid JSONL row' in caplog_evalscope.text
    assert 'Skipping invalid review cache row' in caplog_evalscope.text


def test_filter_prediction_cache_skips_malformed_and_schema_invalid_rows(
    tmp_path: Path,
    caplog_evalscope: pytest.LogCaptureFixture,
) -> None:
    manager = _make_manager(tmp_path)
    dataset = MemoryDataset(
        samples=[
            Sample(id=0, input='question-0', target='answer'),
            Sample(id=1, input='question-1', target='answer'),
        ],
        name='dummy',
    )
    valid_row = ModelResult.from_task_state(_make_task_state(0)).model_dump(mode='json')
    incomplete_row = ModelResult.from_task_state(_make_task_state(1)).model_dump(mode='json')
    incomplete_row['model_output'] = None
    prediction_file = manager.get_prediction_cache_path('default')
    with open(prediction_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(valid_row) + '\n')
        f.write(json.dumps(incomplete_row) + '\n')
        f.write(json.dumps({'legacy_field': 'not a valid ModelResult'}) + '\n')
        f.write('{"index":')

    cached_task_states, filtered_dataset = manager.filter_prediction_cache('default', dataset)

    assert [state.sample_id for state in cached_task_states] == [0]
    assert [sample.id for sample in filtered_dataset.samples] == [1]
    assert 'Skipping invalid JSONL row' in caplog_evalscope.text
    assert 'Skipping invalid prediction cache row' in caplog_evalscope.text


@pytest.mark.parametrize('invalid_tail', [False, True])
def test_filter_prediction_cache_keeps_last_valid_prediction(tmp_path: Path, invalid_tail: bool) -> None:
    manager = _make_manager(tmp_path)
    dataset = MemoryDataset(samples=[Sample(id=i, input=f'question-{i}', target='answer') for i in range(3)])
    for sample_id, answer in [(0, 'first'), (1, 'other'), (0, 'latest')]:
        state = _make_task_state(sample_id)
        state.output = ModelOutput.from_content(model='mock-model', content=answer)
        manager.save_prediction_cache('default', state)
    manager.close()
    if invalid_tail:
        row = ModelResult.from_task_state(_make_task_state(0)).model_dump(mode='json')
        row['model_output'] = None
        with open(manager.get_prediction_cache_path('default'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')

    cached_states, remaining_dataset = manager.filter_prediction_cache('default', dataset)

    assert [state.sample_id for state in cached_states] == [0, 1]
    assert [state.output.completion for state in cached_states] == ['latest', 'other']
    assert [sample.id for sample in remaining_dataset] == [2]


def test_filter_prediction_cache_keeps_distinct_repeat_samples(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    states = [_make_task_state(i, group_id=0) for i in range(2)]
    dataset = MemoryDataset(samples=[Sample(id=i, group_id=0, input='question', target='answer') for i in range(2)])
    for state in states:
        manager.save_prediction_cache('default', state)
    manager.close()

    cached_states, remaining_dataset = manager.filter_prediction_cache('default', dataset)

    assert [state.sample_id for state in cached_states] == [0, 1]
    assert [state.group_id for state in cached_states] == [0, 0]
    assert len(remaining_dataset) == 0
