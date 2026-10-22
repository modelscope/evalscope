# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for MMAU gold mapping and answer-letter parsing."""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

CHOICES = ['dog', 'cat', 'cow', 'horse']


def _adapter():
    return get_benchmark('mmau', TaskConfig(datasets=['mmau']))


def _record(answer: str) -> dict:
    return {
        'question': 'What animal is heard?',
        'choices': CHOICES,
        'answer': answer,
        'audio': {'bytes': b'RIFF0000WAVE'},
        'task': 'sound',
    }


def _score(prediction: str, target: str):
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target=target, choices=CHOICES, metadata={'task': 'sound'}),
        output=ModelOutput.from_content('mock', prediction),
        completed=True,
    )
    return _adapter().calculate_metrics(state).score


def test_unlabeled_test_split_is_not_scored() -> None:
    """lmms-lab/mmau's 9,000-row ``test`` split stores 'NA' as every answer; it must not become gold 'A'."""
    adapter = _adapter()
    assert adapter.subset_list == ['test_mini']
    with pytest.raises(ValueError, match='not one of the choices'):
        adapter.record_to_sample(_record('NA'))
    assert adapter.record_to_sample(_record('Cow')).target == 'C'


@pytest.mark.parametrize('prediction', ['The sound is a dog barking.', 'I cannot tell from this audio.'])
def test_unparsed_answer_is_not_credited_as_a(prediction: str) -> None:
    score = _score(prediction, 'A')
    assert score.extracted_prediction == ''
    assert score.main_value == 0.0


@pytest.mark.parametrize(
    ('prediction', 'expected'),
    [
        ('ANSWER: Dog barking', ''),
        ('ANSWER: a dog barking', ''),
        ('The answer: a cow? No.\nANSWER: B', 'B'),
        ('**Answer**: b.', 'B'),
        ('ANSWER: [D]', 'D'),
    ],
)
def test_answer_line_reads_only_a_choice_letter(prediction: str, expected: str) -> None:
    assert _score(prediction, 'B').extracted_prediction == expected


def test_evaluation_version_bumped_for_scoring_change() -> None:
    assert _adapter()._benchmark_meta.evaluation_version == 'v1.1'
