from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character


@register_benchmark(
    BenchmarkMeta(
        name='boolq',
        pretty_name='BoolQ',
        dataset_id='google/boolq',
        hf_dataset_id='google/boolq',
        default_subset='default',
        subset_list=['default'],
        eval_split='validation',
        train_split='train',
        evaluation_version='v1.1',
        supports_choice=True,
        choice_instructions='Answer the question in `question` using only the accompanying passage.',
        tags=[Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        extra_params={
            'decision_kind': {
                'type': 'str',
                'value': 'choice',
                'description': 'Decision API question type: choice (default) or noul. Requires a decision API for noul.',
            },
            'decision_threshold': {
                'type': 'float',
                'value': 0.5,
                'description': 'Classify Noul as yes when P(yes) is at least this threshold, in [0, 1].',
            },
        },
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
        description="""
## Overview

Binary reading comprehension using a passage and a naturally occurring question.

## Task Description

- **Task Type**: Single-choice classification
- **Input**: Passage and yes/no question
- **Output**: Yes or No
- **Domain**: English reading comprehension

## Key Features

- Public dataset: `google/boolq` on ModelScope
- Preserves the source labels and complete task context
- Supports chat generation and text System One Choice models

## Evaluation Notes

- Uses the public validation labels, zero-shot by default. Decision APIs use two-option Choice by default. Set `extra_params.decision_kind="noul"` for P(yes), with `decision_threshold=0.5` by default.
- Defaults to 0-shot; training examples can be configured where a training split is available
- System prompts become Choice task instructions, without a native chat-role hierarchy
- Evaluation semantics version: v1.1
""",
    )
)
class BoolQAdapter(MultiChoiceAdapter):
    """Evaluate the original public BoolQ validation set as binary Choice."""

    decision_noul_positive_choice = 'Yes'

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        answer = record['answer']
        if not isinstance(answer, bool):
            raise ValueError('BoolQ requires a boolean answer.')
        return Sample(
            input=f'Passage:\n{record["passage"]}\n\nQuestion: {record["question"]}',
            choices=['Yes', 'No'],
            target='A' if answer else 'B',
        )
