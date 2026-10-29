"""ANLS must normalize the Levenshtein distance by the lengths of the *normalized* strings.

The distance is computed on whitespace-collapsed lowercase strings, so the denominator has
to use those same lengths; dividing by the raw lengths made the score depend on how much
surrounding whitespace the reference or prediction happened to carry.
"""
import pytest

from evalscope.metrics.nlp.metrics import ANLS


def test_anls_is_insensitive_to_reference_whitespace() -> None:
    # Both references collapse to 'a b': dist('a b', 'a c') = 1 over length 3 -> 1 - 1/3.
    assert ANLS().apply(['a c'], ['A B']) == pytest.approx([2 / 3])
    assert ANLS().apply(['a c'], ['  A  B  ']) == pytest.approx([2 / 3])


def test_anls_is_insensitive_to_prediction_whitespace() -> None:
    assert ANLS().apply(['a c'], ['a b']) == pytest.approx([2 / 3])
    assert ANLS().apply(['  A  C  '], ['a b']) == pytest.approx([2 / 3])


def test_anls_equal_after_normalization_scores_one() -> None:
    assert ANLS().apply(['Paris'], ['  PARIS  ']) == [1.0]
    assert ANLS().apply(['  paris '], ['paris']) == [1.0]


def test_anls_empty_pair_does_not_divide_by_zero() -> None:
    assert ANLS().apply([''], ['']) == [1.0]


def test_anls_below_threshold_is_zeroed() -> None:
    # dist('abcd', 'ab') = 2 over length 4 -> similarity 0.5 stays at the 0.5 threshold.
    assert ANLS().apply(['ab'], ['abcd']) == [0.5]
    # dist 3 over length 4 -> similarity 0.25 < 0.5 threshold -> 0.
    assert ANLS().apply(['a'], ['abcd']) == [0.0]


@pytest.mark.parametrize('reference', ['42', '3.14', 'true', 'null', '{"value": 42}'])
def test_anls_scores_json_scalar_and_object_reference_text(reference: str) -> None:
    """A literal answer must not be rejected just because it is also valid JSON."""
    assert ANLS()(reference, reference) == 1.0


def test_anls_compares_numeric_reference_text() -> None:
    """Numeric reference text still receives the normal edit-distance penalty."""
    assert ANLS()('43', '42') == 0.5


def test_anls_preserves_serialized_answer_alternatives() -> None:
    """Serialized answer lists continue to select the best matching alternative."""
    assert ANLS()('42', '["forty two", "42"]') == 1.0
    assert ANLS()('42', '"42"') == 1.0
    assert ANLS()('42', '[]') == 0.0
