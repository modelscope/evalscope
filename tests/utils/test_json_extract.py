"""Unit tests for the shared last-JSON-object extractor."""

import pytest

from evalscope.utils.json_extract import extract_last_json_object


@pytest.mark.parametrize(
    'text',
    [
        '{"score": 4}',
        '```json\n{"score": 4}\n```',
        '  {"score": 4}  ',
        'Here is the verdict:\n{"score": 4}\nHope this helps!',
    ],
)
def test_single_object_is_extracted(text: str) -> None:
    assert extract_last_json_object(text) == {'score': 4}


def test_example_object_before_the_answer_is_ignored() -> None:
    assert extract_last_json_object('Example: {"score": 0}. Answer: {"score": 4}') == {'score': 4}


def test_nested_object_is_returned_whole() -> None:
    text = 'Payload: {"images": [{"content": [{"text": "A"}]}]}'

    assert extract_last_json_object(text) == {'images': [{'content': [{'text': 'A'}]}]}


def test_unparseable_trailer_does_not_hide_the_answer() -> None:
    text = '{"score": 4}\nExample of bad output: {"score": }'

    assert extract_last_json_object(text) == {'score': 4}


@pytest.mark.parametrize(
    'text',
    [
        '',
        'no json here',
        '[1, 2]',
        '{"score": 4',
        '{"score": }',
    ],
)
def test_text_without_a_parseable_object_returns_none(text: str) -> None:
    assert extract_last_json_object(text) is None
