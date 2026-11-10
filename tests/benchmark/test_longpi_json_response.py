"""longpi_bench: the model's last JSON object wins when prose contains an earlier one."""

import json

import pytest

from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.longpi_bench.utils import parse_json_response
from evalscope.config import TaskConfig


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


def test_nested_object_is_returned_whole() -> None:
    text = 'Answer: {"result": {"score": 4}}'

    assert parse_json_response(text) == {'result': {'score': 4}}


def test_example_before_a_nested_answer_keeps_the_outer_object() -> None:
    text = 'Example: {"score": 0}. Answer: {"result": {"score": 4}}'

    assert parse_json_response(text) == {'result': {'score': 4}}


def test_evaluation_version_tracks_json_response_extraction() -> None:
    adapter = get_benchmark('longpi_bench', TaskConfig(model='mock', datasets=['longpi_bench']))

    assert adapter.benchmark_meta.evaluation_version == 'v1.1'
