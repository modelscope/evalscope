"""C-Eval demonstrations must keep authored option references aligned after choice shuffling."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from datasets import Dataset as HFDataset

from evalscope.api.dataset.hub import DatasetHub
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


@pytest.mark.parametrize('shuffle_choices', [False, True])
def test_fewshot_explanation_and_evaluation_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shuffle_choices: bool
) -> None:
    example = {
        'id': 0,
        'question': '示例题',
        'A': '甲',
        'B': '乙',
        'C': '丙',
        'D': '丁',
        'answer': 'B',
        'explanation': 'A 是干扰项，B 是正确项。',
    }
    question = {
        **example,
        'id': 1,
        'question': '正式题',
        'A': '正确',
        'B': '错误一',
        'C': '错误二',
        'D': '错误三',
        'answer': 'A',
        'explanation': '',
    }
    @contextmanager
    def fake_load(self: DatasetHub, split: str, subset: str, **kwargs: object) -> Iterator[HFDataset]:
        yield HFDataset.from_list([example if split == 'dev' else question])

    monkeypatch.setattr(DatasetHub, 'load', fake_load)

    adapter = get_benchmark(
        'ceval',
        TaskConfig(
            model='mock',
            datasets=['ceval'],
            seed=42,
            dataset_args={
                'ceval': {
                    'local_path': str(tmp_path),
                    'subset_list': ['high_school_mathematics'],
                    'few_shot_num': 1,
                    'shuffle_choices': shuffle_choices,
                }
            },
        ),
    )
    sample = adapter.load_dataset()['high_school_mathematics'][0]
    prompt = sample.input[-1].content
    assert 'A. 甲\nB. 乙\nC. 丙\nD. 丁\n解析：A 是干扰项，B 是正确项。\n答案：B' in prompt
    assert (sample.choices != [question[letter] for letter in 'ABCD']) == shuffle_choices
    assert sample.choices[ord(sample.target) - ord('A')] == '正确'
    assert f'{sample.target}. 正确' in prompt.split(question['question'])[-1]
