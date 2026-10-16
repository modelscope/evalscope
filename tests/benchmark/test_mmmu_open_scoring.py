"""MMMU open questions are scored like the official MMMU evaluation (eval_open): numbers compare
by value, and a list-valued answer accepts any of its entries."""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def _score(target: str, completion: str, question_type: str = 'open', choices=None) -> float:
    adapter = get_benchmark('mmmu', TaskConfig(model='mock', datasets=['mmmu']))
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target=target, choices=choices, metadata={'question_type': question_type}, id=0),
        output=ModelOutput.from_content('mock', completion),
        completed=True,
    )
    score = adapter.calculate_metrics(state).score
    return score.value[score.main_score_name]


@pytest.mark.parametrize(
    ('target', 'answer', 'expected'),
    [
        ('30.0', '30', 1.0),
        ('9.20', '9.2', 1.0),
        ('2000000', '2,000,000', 1.0),
        ("['Tampa', 'Florida']", 'Tampa', 1.0),
        ("['24/7', '3.429']", '24/7', 1.0),
        ('30.0', '31', 0.0),
        ("['Tampa', 'Florida']", 'Miami', 0.0),
    ],
)
def test_open_question(target, answer, expected):
    assert _score(target, f'Working.\nANSWER: {answer}') == expected


def test_multiple_choice_unchanged():
    choices = ['one', 'two', 'three', 'four']
    assert _score('B', 'ANSWER: B', 'multiple-choice', choices) == 1.0
    assert _score('B', 'ANSWER: C', 'multiple-choice', choices) == 0.0
