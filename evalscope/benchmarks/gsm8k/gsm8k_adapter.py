# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.""".lstrip(
)  # noqa: E501

FEWSHOT_TEMPLATE = """
Here are some examples of how to solve similar problems:

{fewshot}

""".lstrip() + PROMPT_TEMPLATE

# GSM8K Description with standard format
GSM8K_DESCRIPTION = """
## Overview

GSM8K (Grade School Math 8K) is a high-quality dataset of 8.5K linguistically diverse grade school math word problems created by human problem writers. The dataset is specifically designed to evaluate and improve the multi-step mathematical reasoning capabilities of language models.

## Task Description

- **Task Type**: Mathematical Word Problem Solving
- **Input**: Natural language math word problem
- **Output**: Numerical answer derived through step-by-step reasoning
- **Difficulty**: Grade school level (2-8 reasoning steps required)

## Key Features

- Problems require basic arithmetic operations (addition, subtraction, multiplication, division)
- Solutions involve 2 to 8 sequential reasoning steps
- High linguistic diversity in problem formulations
- Human-written problems ensuring natural language quality
- Clear numerical answers for objective evaluation

## Evaluation Notes

- Default configuration uses **4-shot** examples with Chain-of-Thought (CoT) prompting
- Answers should be formatted within `\\boxed{}` for proper extraction
- The metric extracts numerical values for accuracy comparison
- Supports both zero-shot and few-shot evaluation modes
"""


@register_benchmark(
    BenchmarkMeta(
        name='gsm8k',
        pretty_name='GSM8K',
        dataset_id='AI-ModelScope/gsm8k',
        tags=[Tags.MATH, Tags.REASONING],
        description=GSM8K_DESCRIPTION,
        paper_url='https://arxiv.org/abs/2110.14168',
        subset_list=['main'],
        few_shot_num=4,
        train_split='train',
        eval_split='test',
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class GSM8KAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        DELIM = '####'
        question = record['question']
        answer = record['answer'].split(DELIM)
        target = answer.pop().strip()
        reasoning = DELIM.join(answer)

        return Sample(input=question, target=target, metadata={'reasoning': reasoning.strip()})

    def sample_to_fewshot(self, sample: Sample) -> str:
        if sample.metadata:
            return (
                f'{sample.input}\n\nReasoning:\n' + f"{sample.metadata['reasoning']}\n\n"
                + f'ANSWER: \\boxed{{{sample.target}}}'
            )
        else:
            return ''

    def extract_answer(self, prediction: str, task_state: TaskState):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
