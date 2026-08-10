from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.utils.multi_choices import parse_answers, parse_answers_zh

CHOICES = ['first', 'second', 'third', 'fourth']

THINKING_COMPLETION = (
    'We need answer physics. The last line should be of the format ANSWER: [LETTER].\n\n'
    'Energy conservation gives option B.\n\nNeed answer.</think>ANSWER: B'
)


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
    assert parse_answers(_make_state(THINKING_COMPLETION)) == {'B'}


def test_parse_answers_bracketed_letter() -> None:
    assert parse_answers(_make_state('reasoning\nANSWER: [B]')) == {'B'}


TRAILING_SENTENCE = '\nLet me know if you need more.'

WRAPPED_ANSWER_LINES = [
    'ANSWER: B',
    'ANSWER: (B)',
    'ANSWER: [B]',
    'ANSWER: **B**',
    'ANSWER: (B) 300',
    '### Final Answer: **(B) 300**',
]


def test_parse_answers_wrapped_label_does_not_depend_on_trailing_text() -> None:
    """A wrapped label must be parsed, not guessed from the last capital in the reply.

    Regression test: these forms used to reach `_fallback_parse_answer`, which returns the
    last upper-case character of the whole reply. They therefore appeared to work whenever the
    chosen label happened to be that character, and silently broke as soon as any prose
    followed - scoring a correct answer as a miss.
    """
    for answer_line in WRAPPED_ANSWER_LINES:
        assert parse_answers(_make_state(f'reasoning\n{answer_line}')) == {'B'}, answer_line
        assert parse_answers(_make_state(f'reasoning\n{answer_line}{TRAILING_SENTENCE}')) == {'B'}, answer_line


def test_parse_answers_wrapped_multiple_answers() -> None:
    assert parse_answers(_make_state('reasoning\nANSWER: (A, C)'), multiple_correct=True) == {'A', 'C'}


def test_parse_answers_ignores_bracketed_prose() -> None:
    """Only label-shaped bracket contents may be read as an answer."""
    assert parse_answers(_make_state('ANSWER: (see the diagram above)')).isdisjoint(set('ABCD'))


def test_fallback_rejects_letters_outside_the_choice_set() -> None:
    """A guessed letter that is not one of the sample's labels must not be reported.

    It cannot be the model's choice, so recording it invents an answer that never existed;
    reporting nothing scores the same and leaves the review file diagnosable.
    """
    assert parse_answers(_make_state('the shape is Z-like')) == set()


def test_parse_answers_zh_wrapped_label() -> None:
    """The Chinese parser accepts half- and full-width wrappers around the label."""
    for answer_line in ['答案：(B)', '答案：（B）', '答案：**B**', '答案：（B）36 千克']:
        assert parse_answers_zh(_make_state(f'推理过程\n{answer_line}')) == {'B'}, answer_line
        # A trailing English sentence would otherwise hand the fallback an unrelated capital
        assert parse_answers_zh(_make_state(f'推理过程\n{answer_line}\nNote: hope this helps.')) == {'B'}, answer_line


def test_parse_answers_placeholder_only_yields_no_valid_option() -> None:
    completion = 'The last line of your response should be ANSWER: [LETTER] and nothing else.'
    assert parse_answers(_make_state(completion)).isdisjoint({'A', 'B', 'C', 'D'})


def test_completion_argument_overrides_raw_model_output() -> None:
    """An explicit `completion` must be parsed instead of the raw model output.

    Reasoning models can leak a wrong letter into their chain of thought; callers
    pass the filtered text so the discarded reasoning cannot win the match.
    """
    state = _make_state('If I answer ANSWER: A that is wrong.</think>ANSWER: B')
    assert parse_answers(state) == set()
    assert parse_answers(state, completion='ANSWER: B') == {'B'}

    zh_state = _make_state('如果答案：A 就错了。</think>答案：B')
    assert parse_answers_zh(zh_state) == {'A'}
    assert parse_answers_zh(zh_state, completion='答案：B') == {'B'}


def test_configured_filter_reaches_multi_choice_extraction() -> None:
    """A configured filter must affect the extracted answer, not just be computed.

    Regression test: `MultiChoiceAdapter.extract_answer` used to discard the filtered
    prediction and re-read the raw completion, which silently disabled every filter
    (e.g. `remove_until` for stripping reasoning) for all multi-choice benchmarks.
    """
    adapter = get_benchmark(
        'gpqa_diamond',
        TaskConfig(
            datasets=['gpqa_diamond'],
            dataset_args={'gpqa_diamond': {
                'filters': {
                    'remove_until': '</think>'
                }
            }},
        ),
    )
    state = _make_state('If I answer ANSWER: A that is wrong.</think>ANSWER: B')

    assert adapter.filter_ensemble is not None
    assert adapter.filter_prediction(state.output.completion, state) == 'B'
