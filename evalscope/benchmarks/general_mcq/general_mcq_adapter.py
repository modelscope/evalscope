# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# flake8: noqa

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='general_mcq',
        pretty_name='General-MCQ',
        description="""
## Overview

General-MCQ is a customizable multiple-choice question answering benchmark for evaluating language models. It supports flexible data formats and variable number of answer choices.

## Task Description

- **Task Type**: Multiple-Choice Question Answering
- **Input**: Question with 2-10 answer choices (A through J)
- **Output**: Selected answer choice(s)
- **Flexibility**: Supports custom datasets via local files, single or multiple correct answers

## Key Features

- Flexible number of choices (A through J)
- Custom dataset support via local file loading
- Chinese single-/multiple-answer prompt templates (optional CoT variants)
- Configurable few-shot examples
- Accuracy-based evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation with single-answer template
- Set `extra_params.multiple_correct=True` to evaluate questions with multiple correct answers
- Set `extra_params.use_cot=True` to switch to chain-of-thought prompt templates
- Primary metric: **Accuracy**
- Train split: **dev**, Eval split: **val**
- See [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#mcq) for dataset format
""",
        tags=[Tags.MULTIPLE_CHOICE, Tags.CUSTOM],
        dataset_id='general_mcq',
        subset_list=['default'],
        metric_list=['acc'],
        few_shot_num=0,
        train_split='dev',
        eval_split='val',
        prompt_template=MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE,
        extra_params={
            'multiple_correct': {
                'type': 'bool',
                'description': 'Whether the dataset contains questions with multiple correct answers. '
                'When True, switches to the multiple-answer prompt template and parser, '
                'and requires the `answer` field to be a list of letters (e.g., ["A", "C"]).',
                'value': False,
            },
            'use_cot': {
                'type': 'bool',
                'description': 'Whether to use the chain-of-thought (CoT) prompt template variant.',
                'value': False,
            },
        },
    )
)
class GeneralMCQAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

        self.multiple_correct = self.extra_params.get('multiple_correct', False)
        self.use_cot = self.extra_params.get('use_cot', False)

        if self.multiple_correct and self.use_cot:
            self.prompt_template = MultipleChoiceTemplate.CHINESE_MULTIPLE_ANSWER_TEMPLATE_COT
        elif self.multiple_correct:
            self.prompt_template = MultipleChoiceTemplate.CHINESE_MULTIPLE_ANSWER_TEMPLATE
        elif self.use_cot:
            self.prompt_template = MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE_COT
        else:
            self.prompt_template = MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE

    def load_from_disk(self, **kwargs):
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record) -> Sample:
        # Extract choices from the record (A, B, C, D, etc.)
        choices = []
        for choice_key in self.choices:
            if choice_key in record:
                choices.append(record[choice_key])
            else:
                break  # Stop when we reach a choice key that doesn't exist

        answer = record['answer']
        if self.multiple_correct and not isinstance(answer, list):
            raise ValueError(
                f"general_mcq with multiple_correct=True requires 'answer' as a list "
                f"of letters (e.g., ['A', 'C']), got {type(answer).__name__}: {answer!r} "
                f"(id={record.get('id', 'unknown')})."
            )

        return Sample(
            input=record['question'],
            choices=choices,
            target=answer,
            metadata={'id': record.get('id', 'unknown')},
        )
