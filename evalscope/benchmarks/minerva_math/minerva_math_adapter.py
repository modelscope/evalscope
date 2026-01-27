from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='minerva_math',
        pretty_name='Minerva-Math',
        tags=[Tags.MATH, Tags.REASONING],
        description="""
## Overview

Minerva-Math is a benchmark designed to evaluate advanced mathematical and quantitative reasoning capabilities of language models. It consists of 272 challenging problems sourced primarily from MIT OpenCourseWare courses, covering university and graduate-level STEM subjects.

## Task Description

- **Task Type**: Advanced STEM Problem Solving
- **Input**: University/graduate-level mathematical or scientific problem
- **Output**: Step-by-step solution with final answer
- **Difficulty**: University to graduate level

## Key Features

- 272 challenging problems from MIT OpenCourseWare
- Covers advanced subjects: solid-state chemistry, astronomy, differential equations, special relativity
- University and graduate-level difficulty
- Tests deep mathematical and scientific reasoning
- Problems require multi-step quantitative reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\\boxed{}` for proper extraction
- Uses LLM-as-judge for complex answer evaluation
- Problems may require domain-specific knowledge (physics, chemistry, etc.)
- Designed to test the upper limits of model reasoning capabilities
""",
        dataset_id='knoveleng/Minerva-Math',
        subset_list=['default'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        eval_split='train',
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    )
)
class MinervaMathAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._use_llm_judge = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['solution'],
            metadata={
                'type': record['type'],
                'idx': record['idx'],
            },
        )

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
