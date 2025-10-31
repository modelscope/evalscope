# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import defaultdict

from evalscope.api.benchmark import BenchmarkMeta, WOChoiceMultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import csv_to_list, jsonl_to_list
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import WOChoiceMultipleChoiceTemplate

# flake8: noqa

logger = get_logger()

SUBSET_LIST = [
    "MultiChoice",
    "NoneMultiChoice"
]

@register_benchmark(
    BenchmarkMeta(
        name='internal_long_reasoning',
        pretty_name='ILReasoning',
        description='Internal dataset with multiple-choice question answering subset and free form answering subset ',
        tags=[Tags.MULTIPLE_CHOICE, Tags.CUSTOM],
        dataset_id='/home/dzj/evalscope/custom_eval/internal/long-reasoning',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=WOChoiceMultipleChoiceTemplate.MULTIPLE_ANSWER,
    )
)
class ILReasoningAdapter(WOChoiceMultiChoiceAdapter):
    """
    something named internal long reasoning dataset adapter
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    def load_from_disk(self, **kwargs):
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record) -> Sample:
        # Extract choices from the record (A, B, C, D, etc.)

        return Sample(
            input=record['question'],
            target=record['answer'],
            metadata={'id': record.get('id', 'unknown')},
        )
