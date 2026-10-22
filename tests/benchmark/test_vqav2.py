# Copyright (c) Alibaba, Inc. and its affiliates.
"""VQAv2 accuracy must match the official VQA evaluation (GT-Vision-Lab/VQA vqaEval.py)."""
from typing import Any

import pytest

from evalscope.api.benchmark import DataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def _adapter() -> DataAdapter:
    return get_benchmark('vqav2', TaskConfig(datasets=['vqav2']))


def _vqa_score(prediction: str, answers: list) -> float:
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='', metadata={'answers': answers}),
        output=ModelOutput.from_content('mock', f'ANSWER: {prediction}'),
        completed=True,
    )
    return _adapter().calculate_metrics(state).score.value['vqa_score']


@pytest.mark.parametrize(
    ('matches', 'expected'),
    [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9)] + [(matches, 1.0) for matches in range(4, 11)],
)
def test_vqa_score_averages_over_leave_one_out_annotator_sets(matches: int, expected: float) -> None:
    answers = ['two'] * matches + ['three'] * (10 - matches)
    assert _vqa_score('2', answers) == pytest.approx(expected)


@pytest.mark.parametrize(
    'record',
    [
        {'multiple_choice_answer': 'two'},
        {'answer': 'two'},
        {'answers': 'two'},
        {'answers': ['two']},
        {'answers': [{'answer': 'two'}]},
        {'answers': [], 'multiple_choice_answer': 'two'},
    ],
)
@pytest.mark.parametrize(('prediction', 'expected'), [('2', 1 / 3), ('3', 0.0)])
def test_single_reference_preserves_legacy_soft_score(
    record: dict[str, Any], prediction: str, expected: float
) -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample({'question': 'How many?', **record})
    state = TaskState(
        model='mock',
        sample=sample,
        output=ModelOutput.from_content('mock', f'ANSWER: {prediction}'),
        completed=True,
    )
    score = adapter.calculate_metrics(state).score

    assert score.value['vqa_score'] == pytest.approx(expected)
    assert score.value['exact_match'] == float(prediction == '2')


@pytest.mark.parametrize(
    ('prediction', 'answers'),
    [('2', []), ('', ['two'] * 10), ('', ['two']), ('the', ['two'])],
)
def test_missing_answers_or_empty_normalized_prediction_score_zero(prediction: str, answers: list[str]) -> None:
    assert _vqa_score(prediction, answers) == 0.0


def test_evaluation_version_bumped_for_scoring_change() -> None:
    assert _adapter()._benchmark_meta.evaluation_version == 'v1.1'
