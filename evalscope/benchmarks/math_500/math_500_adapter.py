# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='math_500',
        pretty_name='MATH-500',
        tags=[Tags.MATH, Tags.REASONING],
        description="""
## Overview

MATH-500 is a curated subset of 500 problems from the MATH benchmark, designed to evaluate the mathematical reasoning capabilities of language models. It covers five difficulty levels across various mathematical topics including algebra, geometry, number theory, and calculus.

## Task Description

- **Task Type**: Mathematical Problem Solving
- **Input**: Mathematical problem statement
- **Output**: Step-by-step solution with final numerical answer
- **Difficulty Levels**: Level 1 (easiest) to Level 5 (hardest)

## Key Features

- 500 carefully selected problems from the full MATH dataset
- Five difficulty levels for fine-grained evaluation
- Problems cover algebra, geometry, number theory, probability, and more
- Each problem includes a reference solution
- Designed for efficient yet comprehensive math evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\\boxed{}` for proper extraction
- Numeric equivalence checking for answer comparison
- Results can be broken down by difficulty level
- Commonly used for math reasoning benchmarking due to manageable size
""",
        dataset_id='AI-ModelScope/MATH-500',
        subset_list=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    )
)
class Math500Adapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['answer'],
            subset_key=f"Level {record['level']}",
            metadata={
                'question_id': record['unique_id'],
                'solution': record['solution'],
            },
        )

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
