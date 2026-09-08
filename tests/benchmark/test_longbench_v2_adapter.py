import pytest

from evalscope.api.dataset import Sample
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.longbench_v2.longbench_v2_adapter import LongBenchV2Adapter
from evalscope.config import TaskConfig


def make_adapter(few_shot_num: int) -> LongBenchV2Adapter:
    adapter = get_benchmark(
        'longbench_v2',
        TaskConfig(
            datasets=['longbench_v2'],
            dataset_args={'longbench_v2': {'few_shot_num': few_shot_num}},
        ),
    )
    assert isinstance(adapter, LongBenchV2Adapter)
    return adapter


def test_rejects_few_shot_without_training_examples() -> None:
    with pytest.raises(ValueError, match=r'LongBench-v2.*few_shot_num=1.*0-shot'):
        make_adapter(few_shot_num=1)


def test_zero_shot_prompt_keeps_document_question_and_choices() -> None:
    adapter = make_adapter(few_shot_num=0)
    sample = Sample(
        input='Which option is correct?',
        choices=['First', 'Second', 'Third', 'Fourth'],
        target='B',
        metadata={'context': 'Relevant document context.'},
    )

    prompt = adapter.format_prompt_template(sample)

    assert '<text>\nRelevant document context.\n</text>' in prompt
    assert 'Which option is correct?' in prompt
    assert 'A) First\nB) Second\nC) Third\nD) Fourth' in prompt
    assert "'ANSWER: [LETTER]'" in prompt
