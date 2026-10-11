"""MMLU-Redux 'multiple_correct_answers' annotations list option indices separated by commas,
'and', or spaces; every form maps to the annotated options only."""
import pytest

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
