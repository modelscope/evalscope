# evalscope/benchmarks/arxivmath/arxivmath_adapter.py
from __future__ import annotations

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

PROMPT_TEMPLATE = r"""
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.
""".lstrip()

DESCRIPTION = """
## Overview

ArXiv-Math is a benchmark of 103 research-level mathematics problems extracted from arXiv preprints. These problems represent cutting-edge mathematical research and test the ability of language models to reason about advanced mathematical concepts at the frontier of knowledge.

## Task Description

- **Task Type**: Research-Level Mathematics Problem Solving
- **Input**: Advanced mathematical problem from arXiv papers
- **Output**: Step-by-step solution with final answer
- **Difficulty**: Research / graduate level

## Key Features

- 103 problems sourced from arXiv preprints (December 2024 - March 2025)
- Four monthly subsets: december, february, january, march
- Covers diverse areas: algebra, combinatorics, analysis, geometry, number theory
- Problems require deep mathematical reasoning and domain expertise
- Represents the frontier of mathematical research difficulty

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\\boxed{}` for proper extraction
- Numeric accuracy metric with symbolic equivalence checking
- Results can be broken down by monthly competition subset
"""


@register_benchmark(
    BenchmarkMeta(
        name='arxivmath',
        pretty_name='ArXiv-Math',
        dataset_id='evalscope/arxivmath',
        description=DESCRIPTION.strip(),
        tags=[Tags.MATH, Tags.REASONING],
        subset_list=['arxiv/december', 'arxiv/february', 'arxiv/january', 'arxiv/march'],
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
class ArxivMathAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        problem = str(record.get('problem', '')).strip()
        target = str(record.get('answer', '')).strip()
        competition = str(record.get('competition', '')).strip()

        return Sample(
            input=problem,
            target=target,
            subset_key=competition,
            metadata={
                'problem_idx': record.get('problem_idx', None),
                'problem_type': record.get('problem_type', None),
                'source': record.get('source', ''),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        from evalscope.benchmarks.hmmt.utils import extract_hmmt_answer

        return extract_hmmt_answer(prediction)
