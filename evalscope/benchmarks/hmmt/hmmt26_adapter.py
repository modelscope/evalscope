# evalscope/benchmarks/hmmt/hmmt26_adapter.py
from __future__ import annotations

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from .utils import extract_hmmt_answer

PROMPT_TEMPLATE = r"""
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.
""".lstrip()


@register_benchmark(
    BenchmarkMeta(
        name='hmmt26',
        pretty_name='HMMT26',
        dataset_id='evalscope/hmmt_feb_2026',
        description="""
## Overview

HMMT February 2026 is a challenging evaluation benchmark derived from the Harvard-MIT Mathematics Tournament (HMMT) February 2026 competition, one of the most prestigious and difficult high school math contests globally.

## Task Description

- **Task Type**: Competition Mathematics Problem Solving
- **Input**: HMMT-level mathematical problem
- **Output**: Answer with step-by-step reasoning
- **Difficulty**: Advanced high school competition level

## Key Features

- 33 problems from HMMT February 2026 competition
- Four primary domains: Algebra, Combinatorics, Geometry, Number Theory
- Highly challenging competition-level problems
- Tests advanced mathematical reasoning
- Represents elite high school mathematics difficulty

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\\boxed{}` for proper extraction
- Numeric accuracy metric with symbolic equivalence checking
- Problems span multiple mathematical domains
""",
        tags=[Tags.MATH, Tags.REASONING],
        subset_list=['default'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # Dataset only provides 'train' split
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class HMMT26Adapter(DefaultDataAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        problem = str(record.get('problem', '')).strip()
        target = str(record.get('answer', '')).strip()
        ptype = record.get('problem_type', None)

        return Sample(
            input=problem,
            target=target,
            metadata={
                'problem_idx': record.get('problem_idx', None),
                'problem_type': ptype,
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return extract_hmmt_answer(prediction)
