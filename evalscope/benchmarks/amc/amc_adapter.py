from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='amc',
        pretty_name='AMC',
        tags=[Tags.MATH, Tags.REASONING],
        description="""
## Overview

AMC (American Mathematics Competitions) is a benchmark based on problems from the AMC 10/12 competitions from 2022-2024. These multiple-choice problems test mathematical problem-solving skills at the high school level and serve as qualifiers for the AIME competition.

## Task Description

- **Task Type**: Competition Mathematics (Multiple Choice)
- **Input**: AMC-level mathematical problem
- **Output**: Correct answer with step-by-step reasoning
- **Years Covered**: 2022, 2023, 2024

## Key Features

- Problems from AMC 10 and AMC 12 competitions (2022-2024)
- Multiple-choice format with 5 answer options
- Topics: algebra, geometry, number theory, combinatorics
- Difficulty ranges from accessible to challenging
- Official competition problems with verified solutions

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\\boxed{}` for proper extraction
- Three subsets available: `amc22`, `amc23`, `amc24`
- Problems include original URLs for reference
- Solutions available in metadata for verification
""",
        dataset_id='evalscope/amc_22-24',
        subset_list=['amc22', 'amc23', 'amc24'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    )
)
class AMCAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use split as subset
        self.split_as_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['answer'],
            metadata={
                'year': record['year'],
                'url': record['url'],
                'solution': record.get('solution', '')
            },
        )

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
