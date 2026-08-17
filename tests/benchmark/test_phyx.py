"""Unit tests for the PhyX option parser, answer extraction and judge verdict reading.

PhyX ships its options as one quoted string and its answers as free-form physics values, so a
regression in `parse_options`, `extract_*_answer` or `parse_judge_verdict` silently mis-scores
replies instead of raising. The expected prompt/answer strings below are taken verbatim from the
official `PhyX_MC.tsv` / `PhyX_OE.tsv` releases.
"""
from evalscope.benchmarks.phyx.utils import (
    build_mc_question,
    build_oe_question,
    extract_boxed_content,
    extract_mc_answer,
    extract_oe_answer,
    match_mc_answer,
    match_oe_answer,
    parse_judge_verdict,
    parse_options,
)

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


def test_judge_verdict_reads_the_trailing_flag() -> None:
    assert parse_judge_verdict('1')
    assert parse_judge_verdict('Judegement: 1')
    assert not parse_judge_verdict('0')
    assert not parse_judge_verdict('Judegement: 0')


def test_judge_verdict_ignores_digits_inside_discussed_values() -> None:
    """A judge that reasons about '0.49 vs 0.5' must not have those digits read as its verdict."""
    assert parse_judge_verdict('The prediction 0.49 approximates 0.5, so 1')
    assert not parse_judge_verdict('The prediction 1.5 differs from 1.2, so 0')


def test_failed_judge_request_scores_zero() -> None:
    """`LLMJudge.judge` reports failures as an '[ERROR] ...' string containing digits."""
    assert not parse_judge_verdict('[ERROR] Error occurred during qwen3-max@http://host:1 evaluation')
    assert not parse_judge_verdict('')
    assert not parse_judge_verdict('unable to compare')
