"""Unit tests for the SWE-Atlas QnA adapter.

These cover the pieces that run without Docker or a judge model: dataset
conversion, rubric parsing (both polarities), final-answer extraction, and the
reduce step that folds rubric verdicts into the resolve-rate / rubric-pass-rate
metrics. The sandbox run and real judging are exercised end-to-end elsewhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import CaseVerdict, JudgeContext
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.swe_atlas_qna.swe_atlas_qna_adapter import RubricVerdict, SweAtlasQnAAdapter
from evalscope.benchmarks.swe_atlas_qna.utils import (
    extract_final_answer,
    parse_rubric,
    rubric_item_passed,
)
from evalscope.config import TaskConfig


def make_adapter(tmp_path, **extra_params: Any) -> SweAtlasQnAAdapter:
    cfg = TaskConfig(
        datasets=['swe_atlas_qna'],
        dataset_args={'swe_atlas_qna': {'extra_params': extra_params}},
        work_dir=str(tmp_path / 'outputs'),
    )
    adapter = get_benchmark('swe_atlas_qna', cfg)
    assert isinstance(adapter, SweAtlasQnAAdapter)
    return adapter


# ---------------------------------------------------------------------------
# Pure helpers (no sandbox / judge)
# ---------------------------------------------------------------------------


def test_parse_rubric_marks_negative_verifier() -> None:
    items = parse_rubric([
        {'id': 'r1', 'title': 'Mentions the retry loop', 'annotations': {'type': 'positive hli verifier'}},
        {'id': 'r2', 'title': 'Does not invent an API', 'annotations': {'type': 'negative hli verifier'}},
    ])
    assert [item['id'] for item in items] == ['r1', 'r2']
    assert items[0]['negated'] is False
    assert items[1]['negated'] is True


def test_parse_rubric_accepts_json_string_and_defaults_ids() -> None:
    items = parse_rubric('[{"title": "no type key"}]')
    assert len(items) == 1
    assert items[0]['id'] == 'rubric_0'
    assert items[0]['negated'] is False


def test_rubric_item_passed_honours_polarity() -> None:
    # Positive verifier: the claim being met is a pass.
    assert rubric_item_passed(claim_met=True, negated=False) is True
    assert rubric_item_passed(claim_met=False, negated=False) is False
    # Negative verifier: the claim being met is a failure.
    assert rubric_item_passed(claim_met=True, negated=True) is False
    assert rubric_item_passed(claim_met=False, negated=True) is True


def test_extract_final_answer_reads_between_sentinels() -> None:
    text = 'thinking...\n<<FINAL_ANSWER>>\nThe cache is in foo.py.\n<<FINAL_ANSWER>>'
    assert extract_final_answer(text) == 'The cache is in foo.py.'


def test_extract_final_answer_without_tag_returns_text() -> None:
    assert extract_final_answer('  just prose, no sentinel  ') == 'just prose, no sentinel'
    assert extract_final_answer('') == ''


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------


def _record() -> Dict[str, Any]:
    return {
        'task_id': 'kitty-001',
        'prompt': 'Where is the key repeat handled?',
        'reference_answer': 'In keys.c',
        'repository_url': 'https://github.com/kovidgoyal/kitty',
        'repository_base_commit': 'abc123',
        'language': 'c',
        'category': 'codebase_qa',
        'docker_image': 'ghcr.io/scaleapi/swe-atlas:kitty',
        'rubric': '[{"id": "r1", "title": "names keys.c", "annotations": {"type": "positive hli verifier"}}]',
    }


def test_record_to_sample_carries_task_metadata(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    sample = adapter.record_to_sample(_record())
    assert sample.input == 'Where is the key repeat handled?'
    assert sample.metadata['task_id'] == 'kitty-001'
    assert sample.metadata['docker_image'] == 'ghcr.io/scaleapi/swe-atlas:kitty'
    assert sample.metadata['repository_base_commit'] == 'abc123'
    assert sample.metadata['rubric'][0]['id'] == 'r1'
    assert sample.metadata['rubric'][0]['negated'] is False


# ---------------------------------------------------------------------------
# Scoring (reduce)
# ---------------------------------------------------------------------------


def _context(rubric: List[Dict[str, Any]], dirty: Optional[Dict[str, Any]] = None) -> JudgeContext:
    sample = Sample(input='q', metadata={'prompt': 'q', 'rubric': rubric, '_sandbox_state': dirty or {}})
    task_state = TaskState(model='mock', sample=sample, output=None)
    return JudgeContext(task_state=task_state, original_prediction='a', filtered_prediction='a', reference='')


def _verdict(case_id: str, status: str, negated: bool) -> CaseVerdict:
    return CaseVerdict(
        case_id=case_id,
        value=RubricVerdict(status=status, justification=''),
        metadata={'title': case_id, 'negated': negated},
    )


def test_reduce_resolves_when_every_rubric_item_passes(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    rubric = [{'id': 'r1', 'title': 't1', 'negated': False}, {'id': 'r2', 'title': 't2', 'negated': True}]
    definition = adapter.judge_definition(_context(rubric))
    # r1 positive + YES -> pass; r2 negative + NO -> pass. All pass => resolved.
    verdicts = [_verdict('r1', 'YES', False), _verdict('r2', 'NO', True)]
    reduced = definition.reduce_verdicts(verdicts, _context(rubric))
    assert reduced.value['resolved'] == 1.0
    assert reduced.value['rubric_pass_rate'] == 1.0


def test_reduce_partial_pass_is_not_resolved(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    rubric = [{'id': 'r1', 'title': 't1', 'negated': False}, {'id': 'r2', 'title': 't2', 'negated': False}]
    definition = adapter.judge_definition(_context(rubric))
    verdicts = [_verdict('r1', 'YES', False), _verdict('r2', 'NO', False)]
    reduced = definition.reduce_verdicts(verdicts, _context(rubric))
    assert reduced.value['resolved'] == 0.0
    assert reduced.value['rubric_pass_rate'] == 0.5
    assert reduced.metadata['failed_rubric_item_ids'] == ['r2']


def test_reduce_fails_task_when_source_tree_is_dirty(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    rubric = [{'id': 'r1', 'title': 't1', 'negated': False}]
    dirty = {'tree_status': ' M keys.c'}
    definition = adapter.judge_definition(_context(rubric, dirty))
    verdicts = [_verdict('r1', 'YES', False)]
    reduced = definition.reduce_verdicts(verdicts, _context(rubric, dirty))
    # Rubric passed, but the dirty tree forces resolved=0 while keeping the rate.
    assert reduced.value['rubric_pass_rate'] == 1.0
    assert reduced.value['resolved'] == 0.0
    assert reduced.metadata['modified_source'] is True
