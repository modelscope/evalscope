"""Unit tests for the shared whole-word Yes/No verdict extractor."""

import pytest

from evalscope.utils.yes_no import extract_verdict


@pytest.mark.parametrize(
    ('prediction', 'expected'),
    [
        ('Yes', 'YES'),
        ('No', 'NO'),
        ('YES', 'YES'),
        ('NO', 'NO'),
        ('No.', 'NO'),
        ('Yes!', 'YES'),
        ('Yes. There is no mention of it.', 'YES'),
        ('No, nothing contradicts the knowledge.', 'NO'),
    ],
)
def test_single_verdict_is_extracted(prediction: str, expected: str) -> None:
    assert extract_verdict(prediction) == expected


@pytest.mark.parametrize(
    'prediction',
    [
        'Yes and No',
        'No. Actually, Yes.',
        'I do not know.',
        'Maybe.',
        'Yesterday I went out.',
        'I know the answer.',
        'not sure',
        'yes',
        'no',
    ],
)
def test_ambiguous_or_absent_verdict_returns_none(prediction: str) -> None:
    assert extract_verdict(prediction) is None
