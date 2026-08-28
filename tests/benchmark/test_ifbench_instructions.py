import pytest

from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.ifbench.ifbench_adapter import IFBenchAdapter
from evalscope.benchmarks.ifbench.instructions import CharacterCountUniqueWordsChecker


def test_ifbench_evaluation_version_reflects_scoring_change() -> None:
    metadata = BENCHMARK_REGISTRY['ifbench']

    assert metadata.data_adapter is IFBenchAdapter
    assert metadata.evaluation_version == 'v1.1'


@pytest.mark.parametrize(
    'response',
    [
        'Cat. Cat. Cat.',
        'Cat. CAT. Dog.',
        'Cat cat. Dog fox. Owl yak.',
    ],
)
def test_character_count_unique_words_rejects_repeated_words(response: str) -> None:
    checker = CharacterCountUniqueWordsChecker('ratio:sentence_words')

    assert checker.check_following(response) is False


@pytest.mark.parametrize(
    'response',
    [
        'Cat. Dog. Fox.',
        'Cat! Dog? Fox.',
        'Cat, red. Dog, tan. Fox, sky.',
    ],
)
def test_character_count_unique_words_accepts_equal_length_unique_sentences(response: str) -> None:
    checker = CharacterCountUniqueWordsChecker('ratio:sentence_words')

    assert checker.check_following(response) is True


def test_character_count_unique_words_rejects_unequal_sentence_lengths() -> None:
    checker = CharacterCountUniqueWordsChecker('ratio:sentence_words')

    assert checker.check_following('Cat. Longer. Fox.') is False


def test_character_count_unique_words_rejects_punctuation_only_sentences() -> None:
    checker = CharacterCountUniqueWordsChecker('ratio:sentence_words')

    assert checker.check_following('... ... ...') is False


@pytest.mark.parametrize('response', ['Cat. Dog.', 'Cat. Dog. Fox. Owl.'])
def test_character_count_unique_words_requires_three_sentences(response: str) -> None:
    checker = CharacterCountUniqueWordsChecker('ratio:sentence_words')

    assert checker.check_following(response) is False
