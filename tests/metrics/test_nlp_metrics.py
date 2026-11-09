"""ANLS must normalize the Levenshtein distance by the lengths of the *normalized* strings.

The distance is computed on whitespace-collapsed lowercase strings, so the denominator has
to use those same lengths; dividing by the raw lengths made the score depend on how much
surrounding whitespace the reference or prediction happened to carry.
"""
from collections.abc import Callable
from functools import partial

import pytest

from evalscope.benchmarks.general_qa_vqa_metrics import calculate_metric_score
from evalscope.metrics import bleu_ngram_one_sample
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


@pytest.fixture
def bleu_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    import nltk

    # These are single sentences: retain Treebank tokenization without downloading punkt_tab.
    monkeypatch.setattr(nltk, 'word_tokenize', partial(nltk.word_tokenize, preserve_line=True))


@pytest.mark.usefixtures('bleu_tokenizer')
@pytest.mark.parametrize(
    'score_function',
    [bleu_ngram_one_sample, partial(calculate_metric_score, 'BLEU')],
    ids=['helper', 'general_qa_vqa'],
)
class TestCumulativeBLEU:

    @pytest.mark.parametrize(
        ('metric', 'expected'),
        [
            ('bleu-1', 7 / 8),
            ('bleu-2', (5 / 8)**(1 / 2)),
            ('bleu-3', (5 / 16)**(1 / 3)),
            ('bleu-4', (1 / 8)**(1 / 4)),
        ],
    )
    def test_combines_precisions_through_each_ngram_order(
        self, score_function: Callable[[str, str], dict[str, float]], metric: str, expected: float
    ) -> None:
        # Equal lengths give BP=1; the clipped precisions are 7/8, 5/7, 3/6, and 2/5.
        scores = score_function('the cat is on the mat outside today', 'the cat was on the mat outside today')

        assert scores[metric] == pytest.approx(expected)

    def test_identical_sentences_score_one(self, score_function: Callable[[str, str], dict[str, float]]) -> None:
        sentence = 'the cat is on the mat outside today'

        assert score_function(sentence, sentence) == dict.fromkeys(['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'], 1.0)
