# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import os
import re

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
        name='hellaswag',
        pretty_name='HellaSwag',
        tags=[Tags.COMMONSENSE, Tags.MULTIPLE_CHOICE, Tags.KNOWLEDGE],
        description="""
## Overview

HellaSwag is a benchmark for evaluating commonsense natural language inference, specifically testing a model's ability to complete sentences describing everyday situations. The dataset uses adversarial filtering to create challenging distractors that are grammatically correct but semantically implausible.

## Task Description

- **Task Type**: Multiple-Choice Sentence Completion
- **Input**: Context describing an activity or situation
- **Output**: Most plausible continuation from 4 choices (A, B, C, D)
- **Domain**: Everyday activities and commonsense scenarios

## Key Features

- 70,000+ questions testing grounded commonsense inference
- Contexts derived from ActivityNet and WikiHow
- Adversarially-filtered incorrect endings
- Requires understanding of typical event sequences
- Tests physical and social commonsense reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on the validation split
- Endings are preprocessed to clean formatting artifacts
- Context combines `ctx_a` and `ctx_b` fields
- Activity labels available in metadata for analysis
""",
        dataset_id='evalscope/hellaswag',
        metric_list=['acc'],
        subset_list=['default'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class HellaSwagAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        # Preprocess endings
        endings = [self._preprocess(ending) for ending in record['endings']]

        # Create context
        ctx = record['ctx_a'] + ' ' + record['ctx_b'].capitalize()
        context = self._preprocess(ctx)

        # Get target choice letter
        target_letter = ['A', 'B', 'C', 'D'][int(record['label'])]

        return Sample(
            input=context,
            choices=endings,
            target=target_letter,
            metadata={'activity_label': record.get('activity_label', 'unknown')},
        )

    def _preprocess(self, text):
        text = text.strip()
        text = text.replace(' [title]', '. ')
        text = re.sub('\\[.*?\\]', '', text)
        text = text.replace('  ', ' ')
        return text
