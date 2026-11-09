"""job_bench: an example object before the judge's verdict must not break parsing."""

import pytest

from evalscope.benchmarks.job_bench.utils import parse_judge_json


def test_example_object_before_the_verdict_is_ignored():
    content = 'Example: {"score": 0}. Answer: {"score": 4}'

    assert parse_judge_json(content) == {'score': 4}


def test_plain_and_fenced_json_still_parse():
    assert parse_judge_json('{"score": 4}') == {'score': 4}
    assert parse_judge_json('```json\n{"score": 4}\n```') == {'score': 4}


def test_text_without_json_still_raises():
    with pytest.raises(ValueError):
        parse_judge_json('no json here')
