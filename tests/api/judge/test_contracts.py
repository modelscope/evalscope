"""The judge output contract: what a reply must satisfy, and what it must reject.

Every rejection case here was observed in the wild before the refactor and used to produce a
score instead of a parse failure.
"""
from pydantic import BaseModel, Field
from typing import Literal

from evalscope.api.judge import OutputContract


class Grade(BaseModel):
    reasoning: str = ''
    verdict: Literal['A', 'B', 'C']


class Rating(BaseModel):
    reasoning: str = ''
    verdict: float = Field(ge=0.0, le=10.0)


class MultiField(BaseModel):
    extracted_final_answer: str = ''
    reasoning: str = ''
    correct: Literal['yes', 'no']
    confidence: int = Field(default=100, ge=0, le=100)


GRADE = OutputContract(schema_model=Grade)
RATING = OutputContract(schema_model=Rating)
MULTI = OutputContract(schema_model=MultiField)


# ---------------------------------------------------------------------------
# Accepted envelopes
# ---------------------------------------------------------------------------


def test_accepts_a_bare_json_object():
    result = GRADE.parse('{"reasoning": "matches", "verdict": "A"}')

    assert result.ok
    assert result.value.verdict == 'A'


def test_accepts_surrounding_whitespace():
    assert GRADE.parse('\n  {"verdict": "B"}  \n').value.verdict == 'B'


def test_accepts_a_fenced_json_block():
    """Models wrap JSON in a fence even when told not to."""
    result = GRADE.parse('Here is my grade:\n```json\n{"verdict": "C"}\n```\n')

    assert result.value.verdict == 'C'


def test_accepts_a_fence_without_a_language_tag():
    assert GRADE.parse('```\n{"verdict": "A"}\n```').value.verdict == 'A'


def test_omitted_optional_field_takes_its_default():
    assert GRADE.parse('{"verdict": "A"}').value.reasoning == ''


def test_accepts_additional_keys_when_required_keys_exist():
    result = GRADE.parse('{"verdict": "A", "provider_trace": "kept for audit"}')

    assert result.ok
    assert result.value.verdict == 'A'


# ---------------------------------------------------------------------------
# Rejected: no readable payload
# ---------------------------------------------------------------------------


def test_rejects_an_empty_response():
    result = GRADE.parse('')

    assert not result.ok
    assert 'empty' in result.error


def test_rejects_prose_containing_the_verdict_letter():
    """``re.search(r'(A|B|C)')`` used to read the A out of "Answer"."""
    result = GRADE.parse('The Answer is correct, so grade A applies.')

    assert not result.ok
    assert 'no JSON object' in result.error


def test_rejects_a_bare_verdict_token():
    assert not GRADE.parse('A').ok


def test_rejects_truncated_json():
    result = GRADE.parse('{"verdict": "A"')

    assert not result.ok
    assert 'no JSON object' in result.error


def test_rejects_malformed_json_that_looks_like_an_object():
    result = GRADE.parse("{'verdict': 'A'}")

    assert not result.ok
    assert 'not valid JSON' in result.error


def test_rejects_a_json_array():
    result = GRADE.parse('["A", "B"]')

    assert not result.ok


# ---------------------------------------------------------------------------
# Rejected: ambiguous payload
# ---------------------------------------------------------------------------


def test_rejects_two_fenced_objects_that_disagree():
    """Taking the first or last one would be a coin flip."""
    result = GRADE.parse('```json\n{"verdict": "A"}\n```\nOn reflection:\n```json\n{"verdict": "B"}\n```')

    assert not result.ok
    assert 'exactly one' in result.error


def test_rejects_two_fenced_objects_even_when_they_agree():
    result = GRADE.parse('```json\n{"verdict": "A"}\n```\n```json\n{"verdict": "A"}\n```')

    assert not result.ok


# ---------------------------------------------------------------------------
# Rejected: schema violations
# ---------------------------------------------------------------------------


def test_rejects_a_missing_required_field():
    result = GRADE.parse('{"reasoning": "looks right"}')

    assert not result.ok
    assert 'Grade' in result.error


def test_rejects_a_value_outside_the_allowed_set():
    result = GRADE.parse('{"verdict": "maybe"}')

    assert not result.ok


def test_rejects_a_verdict_of_the_wrong_type():
    assert not GRADE.parse('{"verdict": 1}').ok


def test_rejects_a_rating_above_the_scale():
    result = RATING.parse('{"verdict": 42}')

    assert not result.ok
    assert 'Rating' in result.error


def test_rejects_a_rating_below_the_scale():
    assert not RATING.parse('{"verdict": -1}').ok


def test_rejects_a_non_numeric_rating():
    assert not RATING.parse('{"verdict": "high"}').ok


def test_accepts_a_decimal_rating():
    assert RATING.parse('{"verdict": 7.5}').value.verdict == 7.5


def test_accepts_a_numeric_string_rating():
    """Pydantic coerces "8" to 8.0; the bound still applies."""
    assert RATING.parse('{"verdict": "8"}').value.verdict == 8.0


# ---------------------------------------------------------------------------
# Prompt-side instruction
# ---------------------------------------------------------------------------


def test_instruction_lists_the_allowed_verdicts():
    instruction = GRADE.instruction()

    assert '"A" or "B" or "C"' in instruction
    assert '"reasoning"' in instruction
    assert 'single JSON object' in instruction
    assert 'additional keys are allowed' in instruction


def test_instruction_states_the_numeric_bounds():
    instruction = RATING.instruction()

    # Both bounds, not either: a judge told only the floor will happily return 42.
    assert 'a number >= 0.0 and <= 10.0' in instruction


def test_instruction_covers_every_field_of_a_multi_field_schema():
    instruction = MULTI.instruction()

    for name in MultiField.model_fields:
        assert f'"{name}"' in instruction


def test_a_reply_following_the_instruction_parses():
    """The instruction and the parser must not drift apart."""
    reply = '{"extracted_final_answer": "42", "reasoning": "matches", "correct": "yes", "confidence": 90}'

    assert MULTI.parse(reply).ok


def test_accepts_json_after_a_reasoning_block():
    """A reasoning judge emits its thinking before the answer."""
    result = GRADE.parse('<think>The prediction says Marlowe, the gold says Shakespeare.</think>\n{"verdict": "B"}')

    assert result.value.verdict == 'B'


def test_accepts_braces_inside_a_string_field():
    result = GRADE.parse('{"reasoning": "the set {1, 2} matches", "verdict": "A"}')

    assert result.value.verdict == 'A'


def test_accepts_an_escaped_quote_inside_a_string_field():
    result = GRADE.parse('{"reasoning": "the answer is \\"42\\"", "verdict": "A"}')

    assert result.value.verdict == 'A'


def test_rejects_prose_holding_two_objects():
    assert not GRADE.parse('First {"verdict": "A"} then {"verdict": "B"}').ok
