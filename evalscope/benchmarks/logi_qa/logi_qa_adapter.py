# flake8: noqa: E501

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

DESCRIPTION = """
## Overview

LogiQA is a benchmark for evaluating logical reasoning abilities, sourced from expert-written questions originally designed for testing human logical reasoning skills in standardized examinations.

## Task Description

- **Task Type**: Logical Reasoning (Multiple-Choice)
- **Input**: Context passage, question, and 4 answer choices
- **Output**: Correct answer letter (A, B, C, or D)
- **Focus**: Deductive reasoning, logical inference

## Key Features

- Expert-written logical reasoning questions
- Requires formal logical analysis
- Tests various reasoning patterns (syllogisms, conditionals, etc.)
- Questions from standardized examinations
- Challenging for both humans and models

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should follow "ANSWER: [LETTER]" format
- Evaluates on test split
- Simple accuracy metric
"""

PROMPT_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
""".strip()


@register_benchmark(
    BenchmarkMeta(
        name='logi_qa',
        pretty_name='LogiQA',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/logiqa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class LogiQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=f"{record['context']}\n{record['question']}",
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
