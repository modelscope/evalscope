"""MMLU-Indic maps records to letter targets per language/script config and has no few-shot split for `_roman`."""

import pytest

from evalscope.api.benchmark import DataAdapter
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.mmlu_indic.mmlu_indic_adapter import SUBSET_LIST
from evalscope.config import TaskConfig

RECORD = {'question': 'प्रश्न?', 'choices': ['क', 'ख', 'ग', 'घ'], 'answer': 2}


def _adapter(**dataset_args) -> DataAdapter:
    args = {'mmlu_indic': dataset_args} if dataset_args else {}
    return get_benchmark('mmlu_indic', TaskConfig(model='mock', datasets=['mmlu_indic'], dataset_args=args))


def test_configs() -> None:
    assert len(SUBSET_LIST) == 21
    assert SUBSET_LIST[0] == 'en'
    assert sum(subset.endswith('_roman') for subset in SUBSET_LIST) == 10
    assert _adapter().few_shot_num == 0


@pytest.mark.parametrize(
    ('subset', 'language'),
    [('hi', 'Hindi'), ('gu_roman', 'Gujarati (Romanized)'), ('en', 'English'), ('or', 'Odia')],
)
def test_record_to_sample(subset: str, language: str) -> None:
    adapter = _adapter()
    adapter.current_subset_name = subset
    sample = adapter.record_to_sample(RECORD)
    assert sample.input == 'प्रश्न?'
    assert sample.choices == ['क', 'ख', 'ग', 'घ']
    assert sample.target == 'C'
    assert sample.metadata == {'language': language}


def test_three_choice_record_is_kept() -> None:
    adapter = _adapter()
    adapter.current_subset_name = 'en'
    sample = adapter.record_to_sample({'question': 'Q?', 'choices': ['a', 'b', 'c'], 'answer': 2})
    assert sample.choices == ['a', 'b', 'c']
    assert sample.target == 'C'


@pytest.mark.parametrize('answer', [3, -1])
def test_answer_outside_the_choices_is_skipped(answer: int) -> None:
    adapter = _adapter()
    adapter.current_subset_name = 'en'
    assert adapter.record_to_sample({'question': 'Q?', 'choices': ['a', 'b', 'c'], 'answer': answer}) == []


def test_few_shot_on_roman_config_raises() -> None:
    adapter = _adapter(few_shot_num=5)
    with pytest.raises(ValueError, match='no `validation` split'):
        adapter.load_fewshot_subset('hi_roman', data_loader=None)
