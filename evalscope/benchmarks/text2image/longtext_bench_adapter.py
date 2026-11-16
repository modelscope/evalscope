# Copyright (c) Alibaba, Inc. and its affiliates.
"""LongText-Bench adapter: long text rendering benchmark (X-Omni, arXiv:2507.22058)."""

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='longtext_bench',
        pretty_name='LongText-Bench',
        dataset_id='X-Omni/LongText-Bench',
        description="""
## Overview

LongText-Bench (X-Omni, arXiv:2507.22058) evaluates long text rendering with 160 prompts each
for English and Chinese, across 8 text-rich scenes (signboards, labeled objects, printed
materials, webpages, slides, posters, subtitles, dialogues).

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation (text rendering)
- **Input**: Text prompt describing a text-rich scene
- **Output**: Generated image transcribed by a VLM and matched against the ground-truth text

## Evaluation Notes

- The ground truth is the dataset ``text`` field, passed to the scorer as the sample target
- Text Score is a Counter-intersection match ratio: word-level for English, character-level
  for Chinese (set via metric args ``lang``), mirroring the official X-Omni evaluator
- The default transcription backend uses Qwen2.5-VL-7B via transformers; scores degrade to 0
  with an error note when transcription fails
""",
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=['en', 'zh'],
        metric_list=['LongTextScore'],
        primary_metric=MetricSelector(name='text_score', aggregation='mean', dimensions={'scope': 'overall'}),
        few_shot_num=0,
        train_split=None,
        eval_split='test',
    )
)
class LongTextBenchAdapter(Text2ImageAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Records carry prompt_id/prompt/text; the ``text`` field is the rendering ground truth."""
        metadata = dict(record)
        prompt = record.get('prompt', '')
        metadata.update(
            {
                'prompt': prompt,
                'category': record.get('category', ''),
            }
        )
        # The dataset stores the ground truth as a list of rendered text fragments
        # (e.g. ['FRESH BREWS...', 'Locally Roasted...']); the official evaluator joins
        # them with spaces before matching.
        gt_text = record.get('text', '')
        if isinstance(gt_text, list):
            gt_text = ' '.join(str(part) for part in gt_text)
        return Sample(
            input=[ChatMessageUser(content=prompt)],
            # The target doubles as the ground-truth text: the framework's image-pair metric
            # channel passes the sample target to the scorer, which LongTextScore consumes.
            target=gt_text,
            metadata=metadata,
        )
