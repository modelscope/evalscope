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
- **Domain**: Algebra, Combinatorics, Geometry, and Number Theory

## Key Features

- 30 problems from the HMMT November 2025 competition
- Sourced from the MathArena `hmmt_nov_2025` dataset and mirrored on ModelScope
- Highly challenging competition-level problems
- Tests advanced mathematical reasoning
- Represents elite high school mathematics difficulty

## Evaluation Notes

- Default configuration loads `evalscope/hmmt_nov_2025` from ModelScope and evaluates the `train` split
- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\\boxed{}` for proper extraction
- Numeric accuracy uses mathematical equivalence checking for integers, fractions, decimals, and symbolic expressions
- No additional runtime dependencies are required
""",
        tags=[Tags.MATH, Tags.REASONING],
        subset_list=['default'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # Dataset only provides 'train' split
        metric_list=[{'acc': {'numeric': True}}],
        prompt_template=PROMPT_TEMPLATE,
        evaluation_version='v1.0',
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
