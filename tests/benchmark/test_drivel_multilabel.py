# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for reading the label list of a Drivelology multi-label answer."""
import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def _adapter():
    return get_benchmark('drivel_multilabel', TaskConfig(datasets=['drivel_multilabel']))


def _score(reply: str):
    adapter = _adapter()
    sample = adapter.record_to_sample({'text': 'I am not lazy, I am on energy-saving mode.', 'label': ['inversion', 'paradox']})
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content('mock', reply), completed=True)
    return adapter.calculate_metrics(state).score


@pytest.mark.parametrize(
    'reply', ['ANSWER: AD', 'ANSWER: A, D', 'ANSWER: A,D', 'ANSWER: A and D', 'Answer: AD', '**ANSWER:** A, D']
)
def test_listed_labels_are_all_read(reply: str) -> None:
    """The prompt lists the letters as 'A,B,C,D,E'; a reply listing labels that way must keep every label."""
    score = _score(reply)
    assert score.extracted_prediction == 'AD'
    assert score.value['exact_match'] == 1.0
    assert score.value['f1'] == 1.0


def test_unanswered_reply_scores_zero() -> None:
    score = _score('These are inversion and paradox.')
    assert score.extracted_prediction == ''
    assert score.value['f1'] == 0.0


def test_evaluation_version_bumped_for_scoring_change() -> None:
    assert _adapter()._benchmark_meta.evaluation_version == 'v1.1'
