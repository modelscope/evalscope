# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the ReportRef identity contract."""
import pytest

from evalscope.report import ReportRef


def test_key_and_parse_round_trip() -> None:
    ref = ReportRef(run_id='20260811_152001', model_id='qwen-plus')
    assert ref.key == '20260811_152001/qwen-plus'
    assert ReportRef.parse(ref.key) == ref
    assert str(ref) == ref.key


def test_parse_splits_on_the_first_separator() -> None:
    # Everything after the first separator is the model id; a model id never spans two segments in
    # practice, but the split contract keeps the run id unambiguous.
    ref = ReportRef.parse('run/model')
    assert ref.run_id == 'run'
    assert ref.model_id == 'model'


def test_parse_rejects_a_value_without_a_separator() -> None:
    with pytest.raises(ValueError):
        ReportRef.parse('run-only')


@pytest.mark.parametrize('bad', ['', '.', '..'])
def test_rejects_empty_and_dot_segments(bad: str) -> None:
    with pytest.raises(ValueError):
        ReportRef(run_id=bad, model_id='model')
    with pytest.raises(ValueError):
        ReportRef(run_id='run', model_id=bad)


@pytest.mark.parametrize('bad', ['a/b', 'a\\b'])
def test_rejects_path_separators(bad: str) -> None:
    with pytest.raises(ValueError):
        ReportRef(run_id=bad, model_id='model')
    with pytest.raises(ValueError):
        ReportRef(run_id='run', model_id=bad)
