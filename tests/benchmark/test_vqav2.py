# Copyright (c) Alibaba, Inc. and its affiliates.
"""VQAv2 accuracy must match the official VQA evaluation (GT-Vision-Lab/VQA vqaEval.py)."""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def _adapter():
    return get_benchmark('vqav2', TaskConfig(datasets=['vqav2']))


def _vqa_score(prediction: str, answers: list) -> float:
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='', metadata={'answers': answers}),
        output=ModelOutput.from_content('mock', f'ANSWER: {prediction}'),
        completed=True,
    )
    return _adapter().calculate_metrics(state).score.value['vqa_score']


@pytest.mark.parametrize(('matches', 'expected'), [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (10, 1.0)])
def test_vqa_score_averages_over_leave_one_out_annotator_sets(matches: int, expected: float) -> None:
    answers = ['two'] * matches + ['three'] * (10 - matches)
    assert _vqa_score('2', answers) == pytest.approx(expected)


def test_evaluation_version_bumped_for_scoring_change() -> None:
    assert _adapter()._benchmark_meta.evaluation_version == 'v1.1'
