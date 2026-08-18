"""Unit tests for the PhyX option parser, answer extraction and judged scoring.

PhyX ships its options as one quoted string and its answers as free-form physics values, so a
regression in `parse_options` or `extract_*_answer` silently mis-scores replies instead of raising.
The expected prompt/answer strings below are taken verbatim from the official `PhyX_MC.tsv` /
`PhyX_OE.tsv` releases.
"""
import pytest
from typing import Any, List, Optional, Tuple

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.phyx.utils import (
    build_mc_question,
    build_oe_question,
    extract_boxed_content,
    extract_mc_answer,
    extract_oe_answer,
    match_mc_answer,
    match_oe_answer,
    parse_options,
)
from evalscope.config import TaskConfig
from evalscope.constants import JudgeStrategy, ScoreStatus

# Record index 0 of the released test set.
DESCRIPTION = ('A patient with a dislocated shoulder is put into a traction apparatus as shown in figure. '
               'The pulls $\\vec{A}$ and $\\vec{B} must combine to produce an outward traction force of '
               '12.8 N on the patient’s arm.')
QUESTION = 'How large should these pulls be?'
RAW_OPTIONS = 'A:"7.55N",B:"5.55N",C:"7.65N",D:"6.65N"'


def test_options_are_parsed_into_a_label_map() -> None:
    assert parse_options(RAW_OPTIONS) == {'A': '7.55N', 'B': '5.55N', 'C': '7.65N', 'D': '6.65N'}


def test_option_values_ending_in_a_backslash_are_kept_whole() -> None:
    """`ast.literal_eval` cannot read these: the trailing backslash escapes the closing quote.

    Record index 53 of the released set; the official TSV renders the same trailing backslash.
    """
    raw = ('A:"1.54 \\text{ m/s}^2\\",B:"1.84 \\text{ m/s}^2\\",'
           'C:"2.54 \\text{ m/s}^2\\",D:"2.84 \\text{ m/s}^2\\"')
    assert parse_options(raw) == {
        'A': '1.54 \\text{ m/s}^2\\',
        'B': '1.84 \\text{ m/s}^2\\',
        'C': '2.54 \\text{ m/s}^2\\',
        'D': '2.84 \\text{ m/s}^2\\',
    }


def test_option_values_containing_commas_and_apostrophes_are_kept_whole() -> None:
    """Splitting on ',' or "'" truncates LaTeX; record indices 507 and 193 rely on this."""
    assert parse_options('A:"\\( \\frac{v}{c^2} I\'a\'b\'e, \\)",B:"x",C:"y",D:"z"')['A'] == (
        '\\( \\frac{v}{c^2} I\'a\'b\'e, \\)'
    )
    assert parse_options('A: "20\\ \\Omega\\", B: "80\\ \\Omega\\"') == {'A': '20\\ \\Omega\\', 'B': '80\\ \\Omega\\'}


def test_mc_question_matches_the_official_rendering() -> None:
    expected = (f'{DESCRIPTION} {QUESTION}Please directly answer the question and provide the correct '
                'OPTION LETTER ONLY, e.g., A, B, C, D. OPTION: A: 7.55N B: 5.55N C: 7.65N D: 6.65N')
    assert build_mc_question(DESCRIPTION, QUESTION, parse_options(RAW_OPTIONS)) == expected


def test_oe_question_matches_the_official_rendering() -> None:
    expected = f'{DESCRIPTION} {QUESTION} Please answer the question with step by step reasoning.'
    assert build_oe_question(DESCRIPTION, QUESTION) == expected


def test_boxed_content_reads_nested_braces() -> None:
    assert extract_boxed_content('so \\boxed{\\frac{1}{2} \\text{ m}} follows') == '\\frac{1}{2} \\text{ m}'


def test_unterminated_box_yields_no_answer() -> None:
    """A reply truncated inside its box must not be scored on the partial expression."""
    assert extract_boxed_content('the answer is \\boxed{50 \\sqrt{101') is None
    assert extract_boxed_content('no box here') is None


def test_oe_answer_prefers_the_box_and_normalizes_latex() -> None:
    assert extract_oe_answer('Thus T = \\boxed{46300}.') == '46300'
    assert extract_oe_answer('\\boxed{\\dfrac{1}{2}}') == '\\frac{1}{2}'
    assert extract_oe_answer('\\boxed{2\\pi r}') == '23.14 r'


def test_oe_answer_falls_back_to_a_stated_final_answer() -> None:
    assert extract_oe_answer('Step 1 ...\nFinal answer: 7.55 N') == '7.55 N'
    assert extract_oe_answer('The correct answer is: 12 m/s') == '12 m/s'


def test_oe_answer_without_a_marker_is_returned_unchanged() -> None:
    """The official evaluator compares the whole reply when it announces no answer."""
    assert extract_oe_answer('  roughly 7.5 newtons  ') == 'roughly 7.5 newtons'


def test_mc_answer_reads_an_announced_label() -> None:
    assert extract_mc_answer('The correct option is D.') == 'D'
    assert extract_mc_answer('Answer: C') == 'C'
    assert extract_mc_answer('A') == 'A'


def test_mc_answer_reads_an_all_caps_answer_marker() -> None:
    """'ANSWER: C' must not fall through to the raw reply and score 0.

    Upstream enumerates only the lower-case and capitalised announcing words; the all-caps spelling
    is added. It cannot be expressed as `re.IGNORECASE` because that would also let the letter class
    match lower-case text.
    """
    assert extract_mc_answer('ANSWER: C') == 'C'
    assert extract_mc_answer('FINAL OPTION: B') == 'B'


def test_lower_case_prose_does_not_yield_an_invented_letter() -> None:
    """A reply whose reasoning contains 'derived'/'because' must not be read as choice B or D."""
    assert extract_mc_answer('the value is derived from the diagram') == 'the value is derived from the diagram'
    assert extract_mc_answer('answer: derived below, so 22.3 degrees') != 'D'


def test_mc_answer_falls_back_to_the_last_echoed_label() -> None:
    """Replies that restate the option list end with the one they picked."""
    assert extract_mc_answer('Comparing the values, B: 5.55N') == 'B'


def test_oe_match_accepts_equal_and_contained_answers() -> None:
    assert match_oe_answer('7.55N', 'Thus \\boxed{7.55N}', '7.55N')
    assert match_oe_answer('roughly 7.55N', 'roughly 7.55N', '7.55N')
    assert not match_oe_answer('6.65N', 'Thus \\boxed{6.65N}', '7.55N')


def test_mc_match_accepts_the_label_as_printed_or_emphasised() -> None:
    assert match_mc_answer('D', 'The correct option is D.', 'D')
    assert match_mc_answer('the value is 6.65N', 'the value is D: 6.65N', 'D')
    assert match_mc_answer('the value is 6.65N', 'the value is **D**', 'D')
    assert not match_mc_answer('B', 'The correct option is B.', 'D')


class _StubJudge:
    """Records whether the judge was consulted and with which prompt."""

    model_id = 'stub-judge'

    def __init__(self, response: str = '{"verdict": true}') -> None:
        self.response = response
        self.prompts: List[str] = []

    def judge(self, prompt: str = '', system_prompt: Optional[str] = None, messages: Any = None) -> str:
        self.prompts.append(prompt or (messages[-1].content if messages else ''))
        return self.response


def _judged_result(
    name: str,
    prediction: str,
    target: str,
    response: str = '{"verdict": true}',
) -> Tuple[Score, _StubJudge]:
    """Run one prediction through the benchmark's judged scoring path."""
    config = TaskConfig(
        model='mock',
        datasets=[name],
        judge_strategy=JudgeStrategy.LLM,
        judge_model_args={'model_id': 'stub-judge'},
    )
    adapter = get_benchmark(name, config)
    judge = _StubJudge(response)
    adapter.llm_judge = judge
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target=target),
        output=ModelOutput.from_content('mock', prediction),
    )
    extracted = adapter.extract_answer(prediction, state)
    return adapter.llm_match_score(prediction, extracted, target, state), judge


def _judged_score(
    name: str,
    prediction: str,
    target: str,
    response: str = '{"verdict": true}',
) -> Tuple[float, _StubJudge]:
    score, judge = _judged_result(name, prediction, target, response)
    return score.value['acc'], judge


def test_mc_judge_is_not_consulted_when_the_reply_names_a_letter() -> None:
    """Upstream settles a committed letter by string comparison and never pays for a judge call."""
    acc, judge = _judged_score('phyx_mc', 'C', 'C')
    assert (acc, judge.prompts) == (1.0, [])

    acc, judge = _judged_score('phyx_mc', 'B', 'C')
    assert (acc, judge.prompts) == (0.0, [])


def test_mc_judge_arbitrates_only_replies_without_a_letter() -> None:
    """A reply that states the option text instead of its label is handed to the judge."""
    prediction = 'The refracted beam bends to 22.3 degrees from the normal.'
    acc, judge = _judged_score('phyx_mc', prediction, 'C')
    assert acc == 1.0
    assert len(judge.prompts) == 1
    assert 'Ground truth answer: C' in judge.prompts[0]


def test_mc_judged_mode_does_not_credit_a_quoted_option() -> None:
    """A reply committing to A while quoting 'D: ...' must not score.

    Upstream's judged path applies plain equality, not the `D:` / `**D**` fallbacks of rule mode.
    """
    prediction = 'Answer: A. For reference the other choices were D: 6.65N and B: 5.55N.'
    acc, judge = _judged_score('phyx_mc', prediction, 'D')
    assert (acc, judge.prompts) == (0.0, [])


def test_oe_judge_settles_answers_that_do_not_match_literally() -> None:
    acc, judge = _judged_score('phyx_oe', 'Thus \\boxed{50 cm}', '0.5 m')
    assert acc == 1.0
    assert 'Ground truth answer: 0.5 m' in judge.prompts[0]
    assert 'Predicted answer: 50 cm' in judge.prompts[0]

    acc, judge = _judged_score('phyx_oe', 'Thus \\boxed{0.5 m}', '0.5 m')
    assert (acc, judge.prompts) == (1.0, [])  # settled by string equality, no judge call


def test_oe_judge_error_excludes_the_sample() -> None:
    """A failed judge request fails the contract, so the sample is excluded instead of scored 0."""
    score, _ = _judged_result(
        'phyx_oe', 'Thus \\boxed{12 m/s}', '9.8 m/s', response='[ERROR] request failed for model 1'
    )
    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_oe_judge_no_longer_reads_a_bare_flag_out_of_prose() -> None:
    """The old parser read the trailing '1' here as a match."""
    score, _ = _judged_result(
        'phyx_oe', 'Thus \\boxed{12 m/s}', '9.8 m/s', response='The values differ, so the judgement is 1'
    )
    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_partially_parsed_options_are_rejected() -> None:
    """A record whose option string yields fewer than the four labels must not reach the model.

    Presenting an incomplete choice list would silently corrupt the measurement, and skipping the
    record would under-report the domain's problem count.
    """
    adapter = get_benchmark('phyx_mc', TaskConfig(model='mock', datasets=['phyx_mc']))
    adapter.image_root = '/nonexistent'
    record = {
        'index': '7',
        'question': 'q',
        'question_simply': 'd',
        'options': 'A:"7.55N",B:"5.55N"',
        'answer': 'A',
        'image': '7.png',
        'category': 'Mechanics',
        'subfield': 'Statics',
        'reasoning_type': [],
    }
    with pytest.raises(ValueError, match='expected exactly'):
        adapter.record_to_sample(record)

    # A full parse whose answer names a label outside A-D is rejected on the same path.
    record['options'] = 'A:"1",B:"2",C:"3",D:"4"'
    record['answer'] = 'E'
    with pytest.raises(ValueError, match='expected exactly'):
        adapter.record_to_sample(record)
