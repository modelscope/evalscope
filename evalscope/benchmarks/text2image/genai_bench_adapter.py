# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser
from evalscope.api.registry import get_metric, register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='genai_bench',
        pretty_name='GenAI-Bench',
        dataset_id='AI-ModelScope/T2V-Eval-Prompts',
        description="""
## Overview

GenAI-Bench is a comprehensive text-to-image benchmark featuring 1600 prompts designed to evaluate image generation models across diverse categories and complexity levels.

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation
- **Input**: Text prompts with varying complexity (basic and advanced)
- **Output**: Generated images evaluated using VQAScore
- **Size**: 1600 prompts

## Key Features

- Large-scale prompt collection for thorough evaluation
- Categorized prompts (basic vs advanced)
- Uses VQAScore for semantic alignment assessment
- Rich metadata including category tags
- Supports both generation and pre-existing image evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **VQAScore** for semantic alignment
- Evaluates images from the **test** split
- Prompts categorized as 'basic' or 'advanced' based on complexity
- Part of the T2V-Eval-Prompts dataset collection
""",
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=['GenAI-Bench-1600'],
        metric_list=['VQAScore'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
    )
)
class GenAIBenchAdapter(Text2ImageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_from_disk(self, **kwargs):
        if os.path.isfile(self.dataset_id):
            file_name = os.path.basename(self.dataset_id)
            file_without_ext = os.path.splitext(file_name)[0]
            self.subset_list = [file_without_ext]

        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record) -> Sample:
        """Convert a record dictionary to a Sample object."""
        advanced = record['tags'].get('advanced')
        return Sample(
            input=[ChatMessageUser(content=record['prompt'])],
            metadata={
                'id': record['id'],
                'prompt': record['prompt'],
                'category': 'advanced' if advanced else 'basic',
                'tags': record.get('tags', []),
                'image_path': record.get('image_path', ''),  # Optional field for existing image path
            }
        )
