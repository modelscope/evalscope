# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Union

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.metric import MetricSelector
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import get_metric, register_benchmark
from evalscope.constants import Tags
from evalscope.utils.function_utils import thread_safe
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='evalmuse',
        pretty_name='EvalMuse',
        dataset_id='AI-ModelScope/T2V-Eval-Prompts',
        description="""
## Overview

EvalMuse is a text-to-image benchmark that evaluates the quality and semantic alignment of generated images using fine-grained analysis with the FGA-BLIP2Score metric.

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation
- **Input**: Text prompt for image generation
- **Output**: Generated image evaluated for quality and semantic fidelity
- **Metric**: FGA-BLIP2Score (Fine-Grained Analysis with BLIP-2)

## Key Features

- Fine-grained semantic alignment evaluation
- Uses BLIP-2 vision-language model for scoring
- Evaluates both image quality and prompt adherence
- Supports diverse prompt categories
- Objective, reproducible metrics

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Only **FGA_BLIP2Score** metric is supported
- Evaluates images from the **test** split
- Can evaluate pre-generated images or generate new ones
""",
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=['EvalMuse'],
        metric_list=['FGA_BLIP2Score'],
        primary_metric=MetricSelector(name='fga_blip2_score', aggregation='mean', dimensions={'scope': 'overall'}),
        few_shot_num=0,
        train_split=None,
        eval_split='test',
    )
)
class EvalMuseAdapter(Text2ImageAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        metric_entry = self.metric_list[0]
        metric_name = list(metric_entry.keys())[0] if isinstance(metric_entry, dict) else metric_entry
        assert len(self.metric_list) == 1 and metric_name == 'FGA_BLIP2Score', (
            'Only FGA_BLIP2Score is supported for EvalMuse'
        )

    @thread_safe
    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        # Get prediction and prompt from task state
        image_path = task_state.metadata.get('image_path', original_prediction)

        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=image_path,
            prediction=image_path,
        )

        # Calculate scores for each configured metric
        try:
            metric_entry = self.metric_list[0]
            metric_name = list(metric_entry.keys())[0] if isinstance(metric_entry, dict) else metric_entry
            metric_args = self.get_metric_args(metric_name)
            metric_cls = get_metric(metric_name)
            metric_func = metric_cls(**metric_args)
            metric_score = metric_func(image_path, task_state.metadata)[0]
            self._record_metric_result(score, metric_name, metric_score)
        except Exception as e:
            logger.error(f'Error calculating metric {metric_name}: {e}')
            self._record_metric_result(score, metric_name, 0)
            score.metadata[metric_name] = f'error: {str(e)}'

        return score
