# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for HiPhO marking-scheme parsing and judge-response handling."""
from evalscope.benchmarks.hipho.utils import (
    criterion_points,
    extract_boxed_answers,
    is_chinese_exam,
    normalize_marking,
    parse_judge_correct,
    parse_judge_points,
    strip_boxed,
)


def test_criterion_points_english():
    assert criterion_points('Award 0.1 pt if the answer uses Newton\'s law.') == 0.1
    assert criterion_points('Award 1.5 pts for the correct derivation.') == 1.5


def test_criterion_points_chinese():
    assert criterion_points('若正确得 0.5 分，否则得 0 分。') == 0.5


def test_criterion_points_multi_tier_takes_max():
    # Partial-credit tiers: the criterion's maximum is the largest stated value.
    text = 'Award 0.5 pt if fully correct, otherwise award 0.2 pt for partial work.'
    assert criterion_points(text) == 0.5


def test_criterion_points_absent():
    assert criterion_points('No numeric allocation stated here.') == 0.0


def test_normalize_marking_shapes():
    assert normalize_marking(None) == []
    assert normalize_marking([]) == []
    # Flat list of criteria -> single scheme.
    assert normalize_marking(['a', 'b']) == [['a', 'b']]
    # Nested list -> one scheme per official alternative.
    assert normalize_marking([['a', 'b'], ['c']]) == [['a', 'b'], ['c']]


def test_extract_boxed_answers_multiple_and_nested():
    text = r'Final: [\boxed{A}, \boxed{\frac{1}{2}}]'
    assert extract_boxed_answers(text) == ['A', r'\frac{1}{2}']


def test_extract_boxed_answers_none():
    assert extract_boxed_answers('no boxes here') == []


def test_strip_boxed():
    assert strip_boxed(r'\boxed{E}') == 'E'
    # Falls back to the trimmed text when there is no box.
    assert strip_boxed('  42  ') == '42'


def test_parse_judge_points_clamps_to_max():
    assert parse_judge_points('0.1', 0.1) == 0.1
    # A judge that over-reports cannot exceed the criterion allocation.
    assert parse_judge_points('5', 0.2) == 0.2
    # Negative or missing values collapse to zero.
    assert parse_judge_points('-1', 1.0) == 0.0
    assert parse_judge_points('no number', 1.0) == 0.0
    assert parse_judge_points('', 1.0) == 0.0


def test_parse_judge_points_rejects_judge_error():
    # A failed judge request must never be read as a score. LLMJudge.judge returns
    # this string on failure and the model id / endpoint inside it contains digits.
    error = (
        '[ERROR] Error occurred during qwen3-max@https://dashscope.aliyuncs.com/'
        'compatible-mode/v1 LLM judge evaluation: timeout'
    )
    assert parse_judge_points(error, 0.2) == 0.0


def test_parse_judge_correct():
    assert parse_judge_correct('[Correct]') is True
    assert parse_judge_correct('[Incorrect]') is False
    assert parse_judge_correct('Correct') is True
    assert parse_judge_correct('') is False
    assert parse_judge_correct('[ERROR] judge request failed, correct endpoint?') is False


def test_is_chinese_exam():
    assert is_chinese_exam('CPhO_2025') is True
    assert is_chinese_exam('PanMechanics_2024') is True
    assert is_chinese_exam('IPhO_2025') is False
    assert is_chinese_exam('PanPhO_2025') is False
