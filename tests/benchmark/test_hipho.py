# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for HiPhO marking-scheme parsing and judge-response handling."""
import os
import pytest
import tempfile

from evalscope.benchmarks.hipho.hipho_adapter import HiPhOAdapter
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


def test_extract_boxed_answers_ignores_unterminated_box():
    # A reply truncated mid-answer must not yield a partial answer, otherwise the
    # spurious entry shifts the ordered alignment used by answer-level scoring and
    # a correct multi-part answer is graded against the wrong golds.
    assert extract_boxed_answers(r'reasoning \boxed{v = \sqrt{2gh') == []
    assert extract_boxed_answers(r'\boxed{A} then \boxed{B} then \boxed{C') == ['A', 'B']


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


def test_load_figure_rejects_path_outside_dataset():
    # Figure paths come from the dataset record, so they must not be able to read
    # arbitrary files off disk.
    adapter = HiPhOAdapter.__new__(HiPhOAdapter)
    with tempfile.TemporaryDirectory() as tmp:
        adapter.data_root = tmp
        for escaping_ref in ['../../../../etc/passwd', '/etc/passwd', 'a/../../outside.png']:
            with pytest.raises(ValueError):
                adapter._load_figure(escaping_ref)
        # A legitimate relative path inside the root is not rejected; it only
        # warns and returns None because the file does not exist here.
        assert adapter._load_figure(os.path.join('image_question', 'a.png')) is None


def test_load_figure_accepts_symlinked_hub_cache_file():
    # The HuggingFace hub cache symlinks snapshot files to a sibling blobs/
    # directory, so validating the resolved target instead of the reference would
    # reject every legitimate figure on that hub.
    adapter = HiPhOAdapter.__new__(HiPhOAdapter)
    adapter._max_image_bytes = None
    with tempfile.TemporaryDirectory() as tmp:
        blobs = os.path.join(tmp, 'blobs')
        data_root = os.path.join(tmp, 'snapshots', 'rev', 'data')
        os.makedirs(blobs)
        os.makedirs(os.path.join(data_root, 'image_question'))
        blob_path = os.path.join(blobs, 'sha123')
        with open(blob_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
        os.symlink(blob_path, os.path.join(data_root, 'image_question', 'a.png'))

        adapter.data_root = data_root
        assert adapter._load_figure('image_question/a.png').startswith('data:image/png;base64,')
