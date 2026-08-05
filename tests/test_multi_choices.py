from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.utils.multi_choices import parse_answers

CHOICES = ['first', 'second', 'third', 'fourth']


def _make_state(completion: str) -> TaskState:
    sample = Sample(input='question', choices=list(CHOICES), target='B')
    return TaskState(
        model='mock',
        sample=sample,
        output=ModelOutput.from_content(model='mock', content=completion),
    )


def test_parse_answers_plain() -> None:
    assert parse_answers(_make_state('ANSWER: B')) == {'B'}


def test_parse_answers_trailing_period_and_lowercase_keyword() -> None:
    assert parse_answers(_make_state('Some reasoning.\nANSWER: C.')) == {'C'}
    assert parse_answers(_make_state('Some reasoning.\nanswer: D')) == {'D'}


def test_parse_answers_multiple_correct() -> None:
    assert parse_answers(_make_state('reasoning\nANSWER: A,B'), multiple_correct=True) == {'A', 'B'}
    assert parse_answers(_make_state('reasoning\nANSWER: AB'), multiple_correct=True) == {'A', 'B'}


def test_parse_answers_ignores_echoed_prompt_placeholder() -> None:
    """The model may restate the required format before giving the real answer.

    The echoed `ANSWER: [LETTER]` placeholder must not shadow the actual answer,
    even when the answer is not at the start of a line (e.g. it directly follows
    an unpaired `</think>` tag emitted by reasoning models).
    """
    completion = (
        'We need answer physics. The last line should be of the format ANSWER: [LETTER].\n\n'
        'Energy conservation gives option B.\n\nNeed answer.</think>ANSWER: B'
    )
    assert parse_answers(_make_state(completion)) == {'B'}


def test_parse_answers_bracketed_letter() -> None:
    assert parse_answers(_make_state('reasoning\nANSWER: [B]')) == {'B'}


def test_parse_answers_placeholder_only_yields_no_valid_option() -> None:
    completion = 'The last line of your response should be ANSWER: [LETTER] and nothing else.'
    assert parse_answers(_make_state(completion)).isdisjoint({'A', 'B', 'C', 'D'})
