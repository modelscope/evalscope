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
        description=
        'HellaSwag is a benchmark for commonsense reasoning in natural language understanding tasks. It consists of multiple-choice questions where the model must select the most plausible continuation of a given context.',
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
