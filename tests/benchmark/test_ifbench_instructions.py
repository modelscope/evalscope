import pytest

from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.ifbench.ifbench_adapter import IFBenchAdapter
from evalscope.benchmarks.ifbench.instructions import (
    CharacterCountUniqueWordsChecker,
    KeywordsMultipleChecker,
    PersonNameCountChecker,
)


def test_ifbench_evaluation_version_reflects_scoring_change() -> None:
    metadata = BENCHMARK_REGISTRY['ifbench']

    assert metadata.data_adapter is IFBenchAdapter
    assert metadata.evaluation_version == 'v1.2'


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


def _person_name_checker(num_person_names: int) -> PersonNameCountChecker:
    checker = PersonNameCountChecker('count:person_names')
    checker.build_description(N=num_person_names)
    return checker


@pytest.mark.parametrize(
    'response',
    [
        'I flew to Miami for a conference last week.',  # contains "Mia"
        'The Available seats were near Leonardo da Vinci hall.',  # contains "Ava" and "Leo"
    ],
)
def test_person_name_count_ignores_substring_matches(response: str) -> None:
    # The listed names only appear as substrings of longer words, not as standalone
    # names, so the requirement of at least one name must not be satisfied.
    checker = _person_name_checker(num_person_names=1)

    assert checker.check_following(response) is False


def test_person_name_count_accepts_standalone_names() -> None:
    checker = _person_name_checker(num_person_names=2)

    assert checker.check_following('Emma and Liam went to the park.') is True


def _keywords_multiple_checker(*keywords: str) -> KeywordsMultipleChecker:
    checker = KeywordsMultipleChecker('count:keywords_multiple')
    checker.build_description(
        keyword1=keywords[0],
        keyword2=keywords[1],
        keyword3=keywords[2],
        keyword4=keywords[3],
        keyword5=keywords[4],
    )
    return checker


def test_keywords_multiple_ignores_substring_occurrences() -> None:
    # Required counts are (1, 2, 3, 5, 7). "art" appears once as a whole word here,
    # but "start"/"smart"/"apart" would inflate a naive substring count.
    checker = _keywords_multiple_checker('art', 'bb', 'cc', 'dd', 'ee')
    response = (
        'The art gallery had a start and a smart, apart display. '
        'bb bb cc cc cc dd dd dd dd dd ee ee ee ee ee ee ee'
    )

    assert checker.check_following(response) is True


def test_keywords_multiple_still_rejects_wrong_counts() -> None:
    checker = _keywords_multiple_checker('art', 'bb', 'cc', 'dd', 'ee')
    # "art" now appears twice as a whole word, violating the required single occurrence.
    response = (
        'The art of the art gallery. bb bb cc cc cc dd dd dd dd dd ee ee ee ee ee ee ee'
    )

    assert checker.check_following(response) is False
