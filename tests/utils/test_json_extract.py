"""Shared JSON extraction: the last object wins when prose contains an earlier one."""

from evalscope.utils.json_extract import extract_last_json_object


def test_plain_object():
    assert extract_last_json_object('{"score": 4}') == {'score': 4}


def test_example_object_before_the_answer_is_ignored():
    text = 'The schema is {"weaknesses": [...]}. Final answer: {"weaknesses": []}'

    assert extract_last_json_object(text) == {'weaknesses': []}


def test_last_of_two_objects_is_returned():
    assert extract_last_json_object('{"draft": true} then {"score": 4}') == {'score': 4}


def test_braces_inside_a_string_value_are_data():
    text = '{"reasoning": "the set {A, B} is open", "score": 4}'

    assert extract_last_json_object(text) == {'reasoning': 'the set {A, B} is open', 'score': 4}


def test_text_without_an_object_returns_none():
    assert extract_last_json_object('no json here') is None
    assert extract_last_json_object('') is None
