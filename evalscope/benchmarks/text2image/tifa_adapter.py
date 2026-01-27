# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='tifa160',
        pretty_name='TIFA-160',
        dataset_id='AI-ModelScope/T2V-Eval-Prompts',
        description="""
## Overview

TIFA-160 is a text-to-image benchmark with 160 carefully curated prompts designed to evaluate the faithfulness and quality of generated images using automated VQA-based evaluation.

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation
- **Input**: Text prompt for image generation
- **Output**: Generated image evaluated using PickScore metric
- **Size**: 160 prompts

## Key Features

- Compact, high-quality prompt set for efficient evaluation
- Uses PickScore for human preference alignment
- Tests diverse image generation capabilities
- Supports both new generation and pre-existing image evaluation
- Reproducible evaluation pipeline

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **PickScore** for human preference alignment
- Evaluates images from the **test** split
- Part of the T2V-Eval-Prompts dataset collection
""",
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=['TIFA-160'],
        metric_list=['PickScore'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
    )
)
class TIFA_Adapter(Text2ImageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
