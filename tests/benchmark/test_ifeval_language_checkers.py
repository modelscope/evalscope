import langdetect.detector_factory as detector_factory
import pytest
from concurrent.futures import ThreadPoolExecutor
from typing import Set

from evalscope.benchmarks.ifeval import instructions

LOWERCASE_ENGLISH_RESPONSE = '\n\nwhat are the boys holding?'


@pytest.fixture
def cold_langdetect() -> None:
    """Force langdetect back to its un-initialized state.

    The race this module guards against only happens on the very first detection,
    so the process-global factory has to be dropped to reproduce it. Touching the
    private global is deliberate: there is no public API to reset it.
    """
    original_factory = detector_factory._factory
    instructions._langdetect_module = None
    detector_factory._factory = None
    yield
    detector_factory._factory = original_factory
    instructions._langdetect_module = None


def _check_lowercase_english(_: int) -> bool:
    checker = instructions.LowercaseLettersEnglishChecker('lowercase')
    checker.build_description()
    return checker.check_following(LOWERCASE_ENGLISH_RESPONSE)


def test_language_checker_is_deterministic_under_concurrency(cold_langdetect: None) -> None:
    """Identical responses must always score the same, even on a cold parallel start.

    `langdetect` both samples randomly and builds its detector factory lazily without
    a lock, so concurrent first calls used to detect against half-loaded language
    profiles and score identical responses differently.
    """
    with ThreadPoolExecutor(max_workers=16) as pool:
        results: Set[bool] = set(pool.map(_check_lowercase_english, range(128)))

    assert results == {True}, f'language detection is not deterministic: {results}'


def test_detection_seed_is_pinned() -> None:
    langdetect = instructions._import_langdetect()

    assert langdetect.DetectorFactory.seed == 0
    # A short ambiguous string is the classic case where an unpinned seed flips.
    assert len({langdetect.detect('並查看更') for _ in range(50)}) == 1
