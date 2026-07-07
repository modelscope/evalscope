# evalscope/benchmarks/cmath/cmath_adapter.py
from __future__ import annotations

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

PROMPT_TEMPLATE = """
{question}
请一步一步推理，最后将答案放在\\boxed{{}}中。
""".lstrip()

DESCRIPTION = """
## Overview

CMATH is a Chinese elementary school mathematics benchmark containing 1,698 problems across grades 1-6. It evaluates the mathematical reasoning capabilities of language models on Chinese-language math word problems at increasing difficulty levels.

## Task Description

- **Task Type**: Chinese Mathematical Word Problem Solving
- **Input**: Chinese math word problem (elementary school level)
- **Output**: Step-by-step reasoning with numerical answer
- **Difficulty**: Grade 1 (easiest) to Grade 6 (hardest)

## Key Features

- 1,098 test problems + 600 validation problems
- Six grade levels (1-6) for fine-grained difficulty analysis
- Problems are in Chinese, testing language-specific reasoning
- Simple numerical answers (integers or decimals)
- Metadata includes reasoning steps count and digit complexity

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\\boxed{}` for proper extraction
- Numeric accuracy metric for answer comparison
- Results can be broken down by grade level
- Chinese prompt template used by default
"""


@register_benchmark(
    BenchmarkMeta(
        name='cmath',
        pretty_name='CMATH',
        dataset_id='evalscope/cmath',
        description=DESCRIPTION.strip(),
        tags=[Tags.MATH, Tags.REASONING, Tags.CHINESE],
        subset_list=['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class CMathAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = str(record.get('question', '')).strip()
        target = str(record.get('golden', '')).strip()
        grade = record.get('grade', 0)

        return Sample(
            input=question,
            target=target,
            subset_key=f'Grade {grade}',
            metadata={
                'reasoning_step': record.get('reasoning_step', None),
                'num_digits': record.get('num_digits', None),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        from evalscope.metrics.math.parser import extract_answer

        return extract_answer(prediction or '')
