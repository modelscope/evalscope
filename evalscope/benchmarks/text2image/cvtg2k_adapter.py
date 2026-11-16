# Copyright (c) Alibaba, Inc. and its affiliates.
"""CVTG-2K adapter: complex visual text generation benchmark (TextCrafter, arXiv:2503.23461)."""

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = ['CVTG', 'CVTG-Style']


@register_benchmark(
    BenchmarkMeta(
        name='cvtg_2k',
        pretty_name='CVTG-2K',
        dataset_id='dnkdnk/CVTG-2K',
        description="""
## Overview

CVTG-2K is a benchmark of 2,000 prompts for evaluating complex visual text generation
(TextCrafter, arXiv:2503.23461). Prompts contain 2-5 quoted text regions that the generated
image must render; prompts are organized by region count under the CVTG and CVTG-Style splits
(the latter annotates style attributes such as size, color and font).

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation (text rendering)
- **Input**: Text prompt with quoted text regions
- **Output**: Generated image scored with Word Accuracy and word-level NED against the quoted text

## Evaluation Notes

- Ground-truth words are the single-quoted fragments of each prompt, matching the official
  TextCrafter evaluator
- Metrics: cvtg_word_accuracy (exact word match) and cvtg_ned (word-level best-match Levenshtein
  distance, higher is better), both normalized to [0, 1]
- The OCR backend uses PaddleOCR (lang=en); scores degrade to 0 with an error note when the
  image contains no recognizable text
""",
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=SUBSET_LIST,
        metric_list=['CVTGWordAccuracy', 'CVTGNED'],
        primary_metric=MetricSelector(name='cvtg_word_accuracy', aggregation='mean', dimensions={'scope': 'overall'}),
        few_shot_num=0,
        train_split=None,
        eval_split='test',
    )
)
class CVTG2KAdapter(Text2ImageAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """CVTG-2K records carry the prompt with quoted text regions; keep the raw fields."""
        metadata = dict(record)
        prompt = record.get('prompt', '')
        metadata.update({'prompt': prompt, 'category': record.get('benchmark_type', record.get('category', ''))})
        return Sample(
            input=[ChatMessageUser(content=prompt)],
            target='',
            metadata=metadata,
        )
