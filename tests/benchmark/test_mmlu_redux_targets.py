"""MMLU-Redux 'multiple_correct_answers' annotations list option indices separated by commas,
'and', or spaces; every form maps to the annotated options only."""
import pytest

from evalscope.api.dataset.utils import data_to_samples
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


@pytest.mark.parametrize(
    ('correct_answer', 'target'),
    [
        ('0 1 3', ['A', 'B', 'D']),
        ('1 2', ['B', 'C']),
        ('(0, 1)', ['A', 'B']),
        ('2 and 3', ['C', 'D']),
    ],
)
def test_multiple_correct_answers_target(correct_answer, target):
    adapter = get_benchmark('mmlu_redux', TaskConfig(model='mock', datasets=['mmlu_redux']))
    record = {
        'question': 'q',
        'choices': ['a', 'b', 'c', 'd'],
        'answer': 1,
        'error_type': 'multiple_correct_answers',
        'correct_answer': correct_answer,
    }
    assert adapter.record_to_sample(record).target == target


@pytest.mark.parametrize(('error_type', 'annotation'), [
    ('wrong_groundtruth', None),
    ('no_correct_answer', '  '),
    ('multiple_correct_answers', None),
    ('multiple_correct_answers', 'depends on interpretation'),
    ('multiple_correct_answers', 'red, unknown'),
    ('wrong_groundtruth', '-1'),
    ('multiple_correct_answers', '0, 4'),
])
def test_unresolved_correction_does_not_consume_a_sample(error_type: str, annotation: str | None) -> None:
    adapter = get_benchmark('mmlu_redux', TaskConfig(model='mock', datasets=['mmlu_redux']))
    record = dict(question='unresolved', choices=['red', 'blue', 'green', 'yellow'], answer=1,
                  error_type=error_type, correct_answer=annotation)
    good = dict(record, question='corrected', error_type='wrong_groundtruth', correct_answer='C')
    samples = data_to_samples([record, good], adapter.record_to_sample, limit=1)
    assert [(sample.input, sample.target) for sample in samples] == [('corrected', ['C'])]


@pytest.mark.parametrize(('annotation', 'target'), [
    ('A, C', ['A', 'C']),
    ('red, green', ['A', 'C']),
    ('0 1 2 3', list('ABCD')),
    ('0 1, 3', ['A', 'B', 'D']),
])
def test_resolved_multiple_answers(annotation: str, target: list[str]) -> None:
    adapter = get_benchmark('mmlu_redux', TaskConfig(model='mock', datasets=['mmlu_redux']))
    record = dict(question='q', choices=['red', 'blue', 'green', 'yellow'], answer=1,
                  error_type='multiple_correct_answers', correct_answer=annotation)
    assert adapter.record_to_sample(record).target == target
