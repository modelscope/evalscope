"""Choice shuffling must preserve the letter references in MMLU-Pro demonstrations."""

import json
from pathlib import Path

import pytest

from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


@pytest.mark.parametrize('shuffle_choices', [False, True])
def test_fewshot_rationale_and_evaluation_choices(tmp_path: Path, shuffle_choices: bool) -> None:
    example = {
        'question_id': 0,
        'category': 'math',
        'question': 'Which number is even?',
        'options': ['one', 'two', 'three', 'five'],
        'answer': 'B',
        'cot_content': 'A is odd and B is even. The answer is (B).',
    }
    question = {
        **example,
        'question_id': 1,
        'question': 'Which number is zero?',
        'options': ['zero', 'one', 'two', 'three'],
        'answer': 'A',
        'cot_content': '',
    }
    for split, row in [('validation', example), ('test', question)]:
        (tmp_path / f'{split}.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')

    adapter = get_benchmark(
        'mmlu_pro',
        TaskConfig(
            model='mock',
            datasets=['mmlu_pro'],
            seed=42,
            dataset_args={
                'mmlu_pro': {
                    'local_path': str(tmp_path),
                    'subset_list': ['math'],
                    'few_shot_num': 1,
                    'shuffle_choices': shuffle_choices,
                }
            },
        ),
    )
    sample = adapter.load_dataset()['math'][0]
    prompt = sample.input[-1].content
    assert 'A one\nB two\nC three\nD five\nA is odd and B is even. ANSWER: B.' in prompt
    assert (sample.choices != question['options']) == shuffle_choices
    assert sample.choices[ord(sample.target) - ord('A')] == 'zero'
    assert f'{sample.target}) zero' in prompt.split(question['question'])[-1]
