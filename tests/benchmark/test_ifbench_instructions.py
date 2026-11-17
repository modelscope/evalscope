import pytest

from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.ifbench import evaluation_lib
from evalscope.benchmarks.ifbench.ifbench_adapter import IFBenchAdapter
from evalscope.benchmarks.ifbench.instructions import (
    AlphabetLoopChecker,
    CharacterCountUniqueWordsChecker,
    EuropeanCapitalsSortChecker,
    IncludeKeywordChecker,
    KeywordsMultipleChecker,
    KeywordSpecificPositionChecker,
    LastWordFirstNextChecker,
    NGramOverlapChecker,
    PersonNameCountChecker,
    PronounCountChecker,
    QuoteExplanationChecker,
    RepeatSpanChecker,
    SentenceAlphabetChecker,
    SentTypeRatioChecker,
    StopWordPercentageChecker,
    TitleCaseChecker,
    WordsPositionChecker,
)
from evalscope.benchmarks.ifbench.instructions_util import split_into_sentences


def test_ifbench_evaluation_version_reflects_scoring_change() -> None:
    metadata = BENCHMARK_REGISTRY['ifbench']

    assert metadata.data_adapter is IFBenchAdapter
    assert metadata.evaluation_version == 'v1.4'


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


def test_ngram_overlap_uses_word_trigrams() -> None:
    reference = 'the quick brown fox jumps over the lazy dog again and again today'
    checker = NGramOverlapChecker('ratio:overlap')
    checker.build_description(reference_text=reference, percentage=100)

    assert checker.check_following(reference) is True
    assert checker.check_following('completely unrelated wording appears in this line') is False
    assert checker.check_following('too short') is False


def test_character_count_unique_words_normalizes_punctuation() -> None:
    checker = CharacterCountUniqueWordsChecker('ratio:sentence_words')

    assert checker.check_following('Foo-bar baz. Foobar quux. Riddle test.') is False


def test_sentence_alphabet_ignores_leading_punctuation() -> None:
    checker = SentenceAlphabetChecker('custom:sentence_alphabet')
    checker.build_description()
    alphabet_words = [
        'Apple', 'Bears', 'Cats', 'Dogs', 'Eagles', 'Foxes', 'Goats', 'Hawks', 'Ibex', 'Jays', 'Kites', 'Lions',
        'Moose', 'Newts', 'Owls', 'Pigs', 'Quail', 'Rats', 'Seals', 'Toads', 'Urial', 'Voles', 'Wolves', 'Xerus',
        'Yaks', 'Zebras',
    ]
    sentences = [f'{word} appear here now.' for word in alphabet_words]

    assert checker.check_following('"' + sentences[0] + ' ' + ' '.join(sentences[1:])) is True
    assert checker.check_following(' '.join([sentences[1], sentences[0], *sentences[2:]])) is False


def test_words_position_ignores_punctuation_and_case() -> None:
    checker = WordsPositionChecker('words:words_position')
    checker.build_description(keyword='vibrant')

    assert checker.check_following('"The Vibrant sun set over a calm VIBRANT sea!"') is True


def test_repeat_span_uses_word_indices() -> None:
    checker = RepeatSpanChecker('repeat:repeat_span')
    checker.build_description(
        prompt_to_repeat='The walls are solid but the stones are cracked and cold.', n_start=0, n_end=7
    )

    assert checker.check_following('The walls are solid but the stones are') is True
    assert checker.check_following('The walls are solid') is False
    assert checker.check_following('The wall') is False


@pytest.mark.parametrize(
    'response',
    [
        '...',
        '!!!',
        '---',
        '***',
        '😀😀😀',
        '   ',
    ],
)
def test_stop_word_percentage_handles_responses_without_words(response: str) -> None:
    # Responses that contain no word tokens (punctuation/symbol/emoji/whitespace only)
    # used to raise ZeroDivisionError from ``num_stopwords / num_words``. Upstream IFBench
    # guards this by returning False, which we restore here.
    checker = StopWordPercentageChecker('ratio:stop_words')
    checker.build_description(percentage=50)

    assert checker.check_following(response) is False


def test_stop_word_percentage_still_scores_normal_responses() -> None:
    checker = StopWordPercentageChecker('ratio:stop_words')
    checker.build_description(percentage=100)

    # A normal response with words should be scored without raising.
    assert checker.check_following('The quick brown fox jumps over the lazy dog.') is True


@pytest.mark.parametrize(
    'response',
    [
        'This is good.',
        'That works.',
        'Who did this?',
        'Everyone agreed.',
        'None of these.',
    ],
)
def test_pronoun_count_accepts_non_personal_pronouns(response: str) -> None:
    # Upstream IFBench counts demonstrative, interrogative and indefinite pronouns too, so a single
    # "This is good." satisfies count:pronouns with N=1. The local set only held personal, possessive
    # and reflexive pronouns and scored those responses as failures (issue #2065).
    checker = PronounCountChecker('count:pronouns')
    checker.build_description(N=1)

    assert checker.check_following(response) is True


def test_pronoun_count_still_requires_the_requested_number() -> None:
    checker = PronounCountChecker('count:pronouns')
    checker.build_description(N=3)

    assert checker.check_following('He left it with them.') is True  # he, it, them
    assert checker.check_following('He left it there.') is False  # he, it
    assert checker.check_following('He left.') is False


@pytest.mark.parametrize(
    'response',
    [
        '',
        '!!!',
        '??? ...',
    ],
)
def test_alphabet_loop_rejects_responses_without_words(response: str) -> None:
    # ``words[0]`` used to raise IndexError on a wordless response. Upstream returns False.
    checker = AlphabetLoopChecker('start:alphabet_end:letter')
    checker.build_description()

    assert checker.check_following(response) is False


def test_alphabet_loop_still_scores_ordered_words() -> None:
    checker = AlphabetLoopChecker('start:alphabet_end:letter')
    checker.build_description()

    assert checker.check_following('Apple banana cherry') is True
    assert checker.check_following('Banana apple cherry') is False


def test_last_word_first_next_rejects_a_wordless_sentence() -> None:
    # Indexing ``split()[-1]``/``split()[0]`` used to raise IndexError when a sentence had no words.
    checker = LastWordFirstNextChecker('last:first_next')
    checker.build_description()

    assert checker.check_following('Hello. ???') is False
    assert checker.check_following('Cat sat. Sat down.') is True


def test_quote_explanation_accepts_an_empty_response() -> None:
    # ``strip(...)[-1]`` used to raise IndexError on an empty response. Upstream returns True.
    checker = QuoteExplanationChecker('quote:explanation')
    checker.build_description()

    assert checker.check_following('') is True
    assert checker.check_following('Hi') is True


def test_include_keyword_uses_word_boundaries() -> None:
    # Upstream requires a word-boundary match; a substring such as "catalog" must not satisfy "cat".
    checker = IncludeKeywordChecker('sentence:keyword')
    checker.build_description(word='cat', N=1)

    assert checker.check_following('The cat sat on the mat.') is True
    assert checker.check_following('The catalog sat on the mat.') is False
    assert checker.check_following('A catapult fired once.') is False
    assert checker.check_following('the CAT sat.') is True


def test_title_case_rejects_a_lowercase_single_letter_word() -> None:
    # Upstream skips non-alphabetic tokens and requires a single-letter word to be uppercase.
    checker = TitleCaseChecker('format:title_case')
    checker.build_description()

    assert checker.check_following('This Is Title Case.') is True
    assert checker.check_following('A Cat Sat Here.') is True
    assert checker.check_following('a Cat Sat Here.') is False
    assert checker.check_following('This is Title Case.') is False
    assert checker.check_following('...') is True


def test_european_capitals_requires_kyiv() -> None:
    # Upstream PR allenai/IFBench#39 added Kyiv between Brussels and Prague (commit 708c06de / merge cd189a9c).
    checker = EuropeanCapitalsSortChecker('custom:european_capitals_sort')
    checker.build_description()
    official = (
        'Reykjavik, Helsinki, Oslo, Tallinn, Stockholm, Riga, Moscow, Copenhagen, '
        'Vilnius, Minsk, Dublin, Berlin, Amsterdam, Warsaw, London, Brussels, Kyiv, '
        'Prague, Luxembourg, Paris, Vienna, Bratislava, Budapest, Vaduz, Chisinau, '
        'Bern, Ljubljana, Zagreb'
    )
    without_kyiv = (
        'Reykjavik, Helsinki, Oslo, Tallinn, Stockholm, Riga, Moscow, Copenhagen, '
        'Vilnius, Minsk, Dublin, Berlin, Amsterdam, Warsaw, London, Brussels, '
        'Prague, Luxembourg, Paris, Vienna, Bratislava, Budapest, Vaduz, Chisinau, '
        'Bern, Ljubljana, Zagreb'
    )
    assert checker.check_following(official) is True
    assert checker.check_following(without_kyiv) is False


def test_keyword_specific_position_is_case_insensitive() -> None:
    # Upstream compares the word at (n, m) after lowercasing both sides.
    checker = KeywordSpecificPositionChecker('words:keywords_specific_position')
    checker.build_description(keyword='world', n=1, m=2)

    assert checker.check_following('Hello WORLD exists now.') is True
    assert checker.check_following('Hello world exists now.') is True
    assert checker.check_following('Hello there exists now.') is False


def test_pronoun_count_tokenizes_contractions_like_upstream() -> None:
    # Upstream uses nltk.word_tokenize, so "They're" contributes "they". Stripping punctuation
    # first would produce "theyre", which is not in the pronoun set.
    checker = PronounCountChecker('count:pronouns')
    checker.build_description(N=1)

    assert checker.check_following("They're fine.") is True
    assert checker.check_following("Who's there?") is True
    assert checker.check_following('The cat sat.') is False


def test_sentence_split_keeps_period_inside_quotes_like_upstream() -> None:
    # The previous custom splitter rewrote `."` to `".`, which made SentTypeRatioChecker
    # treat a quoted statement as ending in `.` when upstream (nltk.sent_tokenize) did not.
    # Official sample_output for IFBench_test key 163 is the concrete case.
    text = (
        'We call the act of changing the structure of a text while maintaining its meaning '
        '"paraphrasing." Paraphrasing is crucial in various contexts, such as summarizing, '
        'translating, or adapting content to suit different audiences or purposes.\n\n'
        'What is the process involved in paraphrasing?'
    )
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].endswith('"')
    assert sentences[1].endswith('.')
    assert sentences[2].endswith('?')

    checker = SentTypeRatioChecker('ratio:sentence_type')
    checker.build_description()
    # 1 declarative ending in `.` and 1 interrogative; the quoted sentence ends in `"`.
    # The required ratio is 2:1, so this response must fail, matching upstream.
    assert checker.check_following(text) is False


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('', []),
        (' \n\t ', []),
        ('Dr. Smith arrived. He left.', ['Dr. Smith arrived.', 'He left.']),
        ('The value is 3.14. It is positive.', ['The value is 3.14.', 'It is positive.']),
        ('First line\nsecond line.', ['First line\nsecond line.']),
        ('! ? .', ['!', '?', '.']),
    ],
)
def test_sentence_split_preserves_upstream_boundaries(text: str, expected: list[str]) -> None:
    assert split_into_sentences(text) == expected


def test_include_keyword_checks_the_requested_sentence_and_escapes_regex() -> None:
    checker = IncludeKeywordChecker('sentence:keyword')
    checker.build_description(word='cat', N=2)
    assert checker.check_following('Dogs rest. CAT sleeps.') is True
    assert checker.check_following('A cat sleeps. Dogs rest.') is False
    assert checker.check_following('A cat sleeps.') is False

    checker.build_description(word='a.b', N=1)
    assert checker.check_following('An a.b token appears.') is True
    assert checker.check_following('An axb token appears.') is False


def test_keyword_specific_position_checks_both_indices_and_keyword_case() -> None:
    checker = KeywordSpecificPositionChecker('words:keywords_specific_position')
    checker.build_description(keyword='WORLD', n=2, m=2)
    assert checker.check_following('First comes here. Hello world again.') is True
    assert checker.check_following('Hello world again. First comes here.') is False
    assert checker.check_following('First comes here. Hello.') is False
    assert checker.check_following('Hello world again.') is False


def test_pronoun_tokenization_preserves_slashes_and_count_threshold() -> None:
    checker = PronounCountChecker('count:pronouns')
    checker.build_description(N=3)
    assert checker.check_following('she/her/hers') is True
    assert checker.check_following('They they they.') is True
    assert checker.check_following("They're fine.") is False
    checker.build_description(N=4)
    assert checker.check_following('she/her/hers') is False


def test_title_case_skips_tokens_starting_with_digits() -> None:
    checker = TitleCaseChecker('format:title_case')
    checker.build_description()
    assert checker.check_following('123 A Cat.') is True
    assert checker.check_following('123 a Cat.') is False


def test_known_curly_quote_normalization_is_intentionally_preserved() -> None:
    # Known compatibility exception: upstream accepts this response without normalizing curly quotes.
    checker = QuoteExplanationChecker('format:quote_unquote')
    checker.build_description()
    assert checker.check_following('“hello”') is False
    assert checker.check_following('“hello” means a greeting.') is True


def test_known_empty_word_guard_is_preserved_with_nltk_sentences() -> None:
    # Upstream accepts three equal-length punctuation-only sentences; the local word guard rejects them.
    checker = CharacterCountUniqueWordsChecker('ratio:sentence_words')
    checker.build_description()
    assert split_into_sentences('! ? .') == ['!', '?', '.']
    assert checker.check_following('! ? .') is False


def test_last_word_first_next_handles_abbreviations_and_line_breaks() -> None:
    checker = LastWordFirstNextChecker('words:last_first')
    checker.build_description()
    assert checker.check_following('Dr. Smith saw a cat.\nCat slept.') is True
    assert checker.check_following('Dr. Smith saw a cat.\nDog slept.') is False


@pytest.mark.parametrize('response', ['', ' \n\t '])
def test_evaluation_short_circuits_empty_incrementing_word_count(response: str) -> None:
    # Direct check_following still raises on empty input upstream and locally; evaluation rejects it first.
    inp = evaluation_lib.InputExample(
        key=0, instruction_id_list=['sentence:increment'], prompt='', kwargs=[{}]
    )
    assert evaluation_lib.test_instruction_following_strict(inp, response).follow_instruction_list == [False]
    assert evaluation_lib.test_instruction_following_loose(inp, response).follow_instruction_list == [False]
