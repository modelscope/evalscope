import pytest

from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark
from evalscope.benchmarks.bbh.bbh_adapter import FREE_FORM, MULTIPLE_CHOICE, BBHAdapter


def test_bbh_evaluation_version_reflects_target_change() -> None:
    metadata = BENCHMARK_REGISTRY['bbh']

    assert metadata.data_adapter is BBHAdapter
    assert metadata.evaluation_version == 'v1.1'


def _sample(subset: str, target: str):
    adapter = get_benchmark('bbh')
    adapter.current_subset_name = subset
    return adapter, adapter.record_to_sample({'input': 'Question?', 'target': target})


@pytest.mark.parametrize('target', [') ]', ')', '> ) )', '] ) } >'])
def test_dyck_target_keeps_its_brackets(target: str) -> None:
    adapter, sample = _sample('dyck_languages', target)

    assert sample.target == target
    assert sample.metadata['task_type'] == FREE_FORM


def test_dyck_correct_answer_matches_target() -> None:
    adapter, sample = _sample('dyck_languages', ') ]')
    extracted = adapter._extract_ff_answer('So we need ")", "]". So the answer is ) ].')

    assert extracted == sample.target


def test_multiple_choice_target_is_the_bare_letter() -> None:
    _, sample = _sample('date_understanding', '(B)')

    assert sample.target == 'B'
    assert sample.metadata['task_type'] == MULTIPLE_CHOICE
