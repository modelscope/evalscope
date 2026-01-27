# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='hpdv2',
        pretty_name='HPD-v2',
        dataset_id='AI-ModelScope/T2V-Eval-Prompts',
        description="""
## Overview

HPD-v2 (Human Preference Dataset v2) is a text-to-image benchmark that evaluates generated images based on human preferences. It uses the HPSv2.1 score metric trained on large-scale human preference data.

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation
- **Input**: Text prompt for image generation
- **Output**: Generated image evaluated against human preferences
- **Metric**: HPSv2.1 Score

## Key Features

- Human preference-aligned evaluation metric
- Trained on large-scale human preference data
- Tests aesthetic quality and prompt alignment
- Supports diverse prompt categories
- Objective, reproducible scoring

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- HPSv2.1 Score metric measures human preference alignment
- Supports local prompt files
- Category tags available in metadata
- Can evaluate existing images or generate new ones
""",
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=['HPDv2'],
        metric_list=['HPSv2.1Score'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
    )
)
class HPDv2Adapter(Text2ImageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_from_disk(self, **kwargs):
        if os.path.isfile(self.dataset_id):
            file_name = os.path.basename(self.dataset_id)
            file_without_ext = os.path.splitext(file_name)[0]
            self.subset_list = [file_without_ext]

        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record):
        return Sample(
            input=[ChatMessageUser(content=record['prompt'])],
            metadata={
                'id': record['id'],
                'prompt': record['prompt'],
                'category': record.get('tags', {}).get('category', ''),
                'tags': record.get('tags', {}),
                'image_path': record.get('image_path', ''),  # Optional field for existing image path
            }
        )
