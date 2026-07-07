# evalscope/benchmarks/imo_answerbench/imo_answerbench_adapter.py
from __future__ import annotations

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = r"""
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.
""".lstrip()

DESCRIPTION = """
## Overview

IMO-AnswerBench is a benchmark of 400 challenging problems sourced from the International Mathematical Olympiad (IMO) Shortlists. It covers four major mathematical domains and is designed to evaluate advanced mathematical reasoning capabilities of language models at the olympiad level.

## Task Description

- **Task Type**: Olympiad Mathematics Problem Solving
- **Input**: IMO Shortlist mathematical problem
- **Output**: Step-by-step solution with final answer
- **Difficulty**: International olympiad level

## Key Features

- 400 problems from IMO Shortlists (2005-2024)
- Four domains: Algebra, Combinatorics, Geometry, Number Theory
- Subcategories include: Operation, Inequality, Sequence, Polynomial, Functional Equation, etc.
- Answers range from simple integers to complex LaTeX expressions (intervals, sets, fractions)
- Represents the highest difficulty level in mathematical problem-solving benchmarks

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\\boxed{}` for proper extraction
- Uses numeric equivalence checking with LLM-as-judge for complex answers
- Results can be broken down by Category (Algebra, Combinatorics, Geometry, Number Theory)
- Many answers involve symbolic expressions requiring mathematical equivalence checking
"""


@register_benchmark(
    BenchmarkMeta(
        name='imo_answerbench',
        pretty_name='IMO-AnswerBench',
        dataset_id='evalscope/imo-answerbench',
        description=DESCRIPTION.strip(),
        tags=[Tags.MATH, Tags.REASONING],
        subset_list=['Algebra', 'Combinatorics', 'Geometry', 'Number theory'],
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
class IMOAnswerBenchAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._use_llm_judge = True
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        problem = str(record.get('Problem', '')).strip()
        target = str(record.get('Short Answer', '')).strip()
        category = str(record.get('Category', '')).strip()

        return Sample(
            input=problem,
            target=target,
            subset_key=category,
            metadata={
                'problem_id': record.get('Problem ID', ''),
                'subcategory': record.get('Subcategory', ''),
                'source': record.get('Source', ''),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract final answer using math parser with fallback to boxed extraction."""
        from evalscope.metrics.math.parser import extract_answer

        return extract_answer(prediction or '')
