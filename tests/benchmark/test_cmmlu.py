from unittest.mock import patch

from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def test_cmmlu_few_shot_uses_dev_split() -> None:
    config = TaskConfig(
        datasets=['cmmlu'],
        dataset_args={'cmmlu': {
            'few_shot_num': 5,
            'subset_list': ['agronomy'],
        }},
    )
    adapter = get_benchmark('cmmlu', config)
    test_dataset = DatasetDict({
        'agronomy':
        MemoryDataset([
            Sample(input='test question', choices=['A', 'B', 'C', 'D'], target='A', subset_key='agronomy')
        ])
    })
    fewshot_dataset = DatasetDict({
        'agronomy':
        MemoryDataset([
            Sample(
                input=f'dev question {index}',
                choices=['A', 'B', 'C', 'D'],
                target='A',
                subset_key='agronomy',
            ) for index in range(5)
        ])
    })

    assert adapter.train_split == 'dev'
    assert adapter._should_load_fewshot()

    with patch.object(adapter, 'load', return_value=(test_dataset, fewshot_dataset)):
        loaded = adapter.load_dataset()

    prompt = loaded['agronomy'][0].input[-1].content
    assert prompt.startswith('以下是一些示例问题：')
    assert prompt.count('dev question') == 5
    assert prompt.count('答案：A') == 5
    assert 'ANSWER:' not in prompt
    assert 'test question' in prompt
    assert '{fewshot}' not in prompt
