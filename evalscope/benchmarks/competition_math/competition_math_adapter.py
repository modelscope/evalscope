# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

PROMPT_TEMPLATE = """
Problem:
{question}

Please reason step by step, and put your final answer within \\boxed{{}}.
""".lstrip()

FEWSHOT_TEMPLATE = """
Here are some examples of how to solve similar problems:

{fewshot}
""".lstrip() + PROMPT_TEMPLATE


@register_benchmark(
    BenchmarkMeta(
        name='competition_math',
        pretty_name='Competition-MATH',
        tags=[Tags.MATH, Tags.REASONING],
        description="""
## Overview

Competition-MATH is a comprehensive benchmark of 12,500 challenging competition mathematics problems collected from AMC, AIME, and other prestigious math competitions. It is designed to evaluate the advanced mathematical reasoning capabilities of language models.

## Task Description

- **Task Type**: Competition Mathematics Problem Solving
- **Input**: Mathematical problem from competitions
- **Output**: Step-by-step solution with final answer in \\boxed{}
- **Difficulty Levels**: Level 1 (easiest) to Level 5 (hardest)

## Key Features

- 12,500 problems from mathematics competitions
- Five difficulty levels for comprehensive evaluation
- Topics: Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus
- Each problem includes human-written solutions
- Designed for evaluating advanced mathematical reasoning

## Evaluation Notes

- Default configuration uses **4-shot** examples with Chain-of-Thought prompting
- Answers are extracted from `\\boxed{}` format
- Numeric equivalence checking for answer comparison
- Results can be analyzed by level and problem type
- Uses math_parser for robust answer extraction
""",
        dataset_id='evalscope/competition_math',
        subset_list=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        few_shot_num=4,
        train_split='train',
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class CompetitionMathAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        from evalscope.metrics.math_parser import extract_answer

        return Sample(
            input=record['problem'],
            target=extract_answer(record['solution']),
            subset_key=record['level'],
            metadata={
                'reasoning': record.get('solution', ''),
                'type': record.get('type', ''),
            },
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        return f'Problem:\n{sample.input}\nSolution:\n{sample.target}'

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
