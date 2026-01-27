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
- **Output**: Selected answer choice
- **Flexibility**: Supports custom datasets via local files

## Key Features

- Flexible number of choices (A through J)
- Custom dataset support via local file loading
- Chinese single-answer prompt template
- Configurable few-shot examples
- Accuracy-based evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
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
    )
)
class GeneralMCQAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

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

        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],
            metadata={'id': record.get('id', 'unknown')},
        )
