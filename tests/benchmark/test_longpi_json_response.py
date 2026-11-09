"""longpi_bench: the model's last JSON object wins when prose contains an earlier one."""

import json

import pytest

from evalscope.benchmarks.longpi_bench.utils import parse_json_response


def test_example_object_before_the_answer_is_ignored():
    text = 'The schema is {"weaknesses": [...]}. Final answer: {"weaknesses": []}'

    assert parse_json_response(text) == {'weaknesses': []}


def test_last_of_two_objects_is_returned():
    text = '{"draft": true} then {"weaknesses": []}'

    assert parse_json_response(text) == {'weaknesses': []}


def test_a_single_object_still_parses():
    assert parse_json_response('{"weaknesses": []}') == {'weaknesses': []}


def test_text_without_json_still_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json_response('no json here')
