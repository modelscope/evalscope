"""Regression tests for TIR-Bench answer scoring."""

import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

PROMPT = 'What is the reading on the gauge?\nA. 10\nB. 20\nC. 30\nD. 40'


def _score(prediction: str, target: str, prompt: str = PROMPT, task: str = 'instrument') -> Score:
    adapter = get_benchmark('tir_bench', config=TaskConfig(datasets=['tir_bench'], judge={'strategy': 'rule'}))
    state = TaskState(
        model='offline',
        sample=adapter.record_to_sample({'prompt': prompt, 'answer': target, 'task': task}),
        output=ModelOutput.from_content('offline', prediction),
        completed=True,
    )
    return adapter.calculate_metrics(state).score


@pytest.mark.parametrize(
    ('prediction', 'target'),
    [
        ('I double-checked the dial markings.\nANSWER: C', 'C'),
        ('Answer: A looks close, but the needle is past 25.\nAnswer: C', 'C'),
        ('**Answer:** C.', 'C'),
        ('Answer: $C$', 'C'),
        ('C', 'C'),
        ('(C)', 'C'),
        (r'Answer: \boxed{C}.', 'C'),
    ],
)
def test_explicit_single_choice_is_scored(prediction: str, target: str) -> None:
    score = _score(prediction, target)

    assert score.extracted_prediction == target
    assert score.main_value == 1


@pytest.mark.parametrize(
    ('prediction', 'target'),
    [
        ('I ruled out option B.', 'B'),
        ('I ruled out option B. I cannot determine the reading.', 'B'),
        ('Option B is wrong.', 'B'),
        ('The correct option is C. I double-checked the dial markings.', 'C'),
        ('Option B matches the needle position. I am fairly confident.', 'B'),
        ('The needle points at 30, so C.', 'C'),
        ('I do not think the answer is B.', 'B'),
        ('I would not answer: B', 'B'),
        ('Answer: B\nAfter checking, I rule out B.', 'B'),
        ('I ruled out option B. My final choice is H.', 'B'),
        ('Answer: C or B', 'C'),
        ('Answer: I think it might be C.', 'C'),
        ('It is unreadable.', 'A'),
        ('', 'A'),
    ],
)
def test_single_choice_does_not_credit_mentions_or_guesses(prediction: str, target: str) -> None:
    assert _score(prediction, target).main_value == 0


@pytest.mark.parametrize(
    ('prompt', 'prediction', 'target', 'expected'),
    [
        (PROMPT, '20', 'B', 1),
        (PROMPT, 'Answer: 20', 'B', 1),
        ('A 200-205\nB 205-210\nC 210-215', '205-210', 'B', 1),
        ('A: 65%\nB: 73%\nC: 81%', '73%', 'B', 1),
        ('A.12\nB.13\nC.14', '13', 'B', 1),
        ('Choose one. (A) Hard to tell (B) Yes (C) No', 'Yes', 'B', 1),
        ('A. 200\nB. 20', '20', 'A', 0),
        ('A. 20\nB. 20', '20', 'A', 0),
        (PROMPT, '2', 'B', 0),
    ],
)
def test_single_choice_matches_only_unique_complete_option_text(
    prompt: str, prediction: str, target: str, expected: int
) -> None:
    assert _score(prediction, target, prompt).main_value == expected


@pytest.mark.parametrize(
    ('task', 'prediction', 'target'),
    [('instrument', 'BA', 'AB'), ('instrument', '20', '20'), ('ocr', 'The text reads ABC.', 'ABC')],
)
def test_other_answer_types_keep_their_scoring(task: str, prediction: str, target: str) -> None:
    assert _score(prediction, target, task=task).main_value == 1


@pytest.mark.parametrize(
    ('task', 'target', 'expects_answer_line'),
    [('instrument', 'C', True), ('instrument', 'AB', False), ('instrument', '20', False), ('ocr', 'A', False)],
)
def test_only_single_choice_prompt_requests_an_answer_line(task: str, target: str, expects_answer_line: bool) -> None:
    adapter = get_benchmark('tir_bench', config=TaskConfig(datasets=['tir_bench']))
    sample = adapter.record_to_sample({'prompt': PROMPT, 'answer': target, 'task': task})
    text = sample.input[0].content[0].text

    if expects_answer_line:
        assert text.startswith(PROMPT)
        assert 'End your response with ANSWER: [LETTER]' in text
    else:
        assert text == PROMPT
