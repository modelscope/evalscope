# evalscope/benchmarks/hmmt/hmmt_nov25_adapter.py
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


# https://huggingface.co/datasets/MathArena/hmmt_nov_2025
@register_benchmark(
    BenchmarkMeta(
        name='hmmt_nov25',
        pretty_name='HMMT-Nov-2025',
        dataset_id='evalscope/hmmt_nov_2025',
        description="""
## Overview

HMMT November 2025 (MathArena) is a challenging evaluation benchmark derived from the Harvard-MIT Mathematics Tournament (HMMT) November 2025 competition, one of the most prestigious and difficult high school math contests globally. It is a different contest from HMMT February 2025 (`hmmt25`).

## Task Description

- **Task Type**: Competition Mathematics Problem Solving
- **Input**: HMMT-level mathematical problem
- **Output**: Answer with step-by-step reasoning
- **Difficulty**: Advanced high school competition level

## Key Features

- 30 problems from HMMT November 2025 competition
- Four primary domains: Algebra, Combinatorics, Geometry, Number Theory
- Highly challenging competition-level problems
- Tests advanced mathematical reasoning
- Represents elite high school mathematics difficulty

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\\boxed{}` for proper extraction
- Numeric accuracy metric
- Problems span multiple mathematical domains
""",
        tags=[Tags.MATH, Tags.REASONING],
        subset_list=['default'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # HF dataset provides split 'train'
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class HMMTNov25Adapter(DefaultDataAdapter):

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
