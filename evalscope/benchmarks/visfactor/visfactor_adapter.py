# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import aggregate_item_accuracy, render_question, score_prediction

DESCRIPTION = """
## Overview

VisFactor evaluates foundational visual cognition in multimodal large language models using 20 vision-centric subtests adapted from the Factor-Referenced Cognitive Test (FRCT). It isolates abilities that support higher-level visual reasoning instead of measuring performance on a single downstream task.

## Task Description

- **Task Type**: Visual cognition assessment with binary and short free-form questions
- **Input**: One to four images interleaved with a task-specific instruction
- **Output**: A JSON object containing a boolean, word, number, coordinate pair, or letter answer
- **Domain**: Visualization and spatial processing, perceptual closure, visual memory, and reasoning

## Key Features

- Contains 3,046 rows representing 808 test items across 20 FRCT subtests
- Uses rule-based variants and grouped consistency checks to reduce average chance performance to approximately 2.9%
- Preserves the official zero-shot prompts and their image ordering from the VLMEvalKit implementation
- Covers hidden figures, gestalt completion, visual memory, mental rotation, path finding, paper folding, and related abilities

## Evaluation Notes

- Uses the **test** split from the ModelScope mirror of the official `VisFactor.tsv`
- Extracts the last `{"answer": ...}` object and applies the official category-specific normalization rules
- A logical test item may contain multiple rows and receives credit only when every row is correct
- Reports each subtest's item-level accuracy; the primary score is the unweighted macro-average over represented subtests
- Scoring is deterministic and does not require an LLM judge
"""  # noqa: E501

_IMAGE_PATTERN = re.compile(r'<IMAGE_(\d+)>')


@register_benchmark(
    BenchmarkMeta(
        name='visfactor',
        pretty_name='VisFactor',
        dataset_id='lmms-lab-encoder/visfactor',
        tags=[Tags.MULTI_MODAL, Tags.REASONING, Tags.QA],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2502.16435',
        metric_list=['accuracy'],
        primary_metric=MetricSelector(
            name='accuracy',
            aggregation='macro_mean',
            dimensions={'scope': 'categories'},
        ),
        eval_split='test',
        evaluation_version='v1.0',
    )
)
class VisFactorAdapter(VisionLanguageAdapter):
    """Adapter for the official VisFactor prompts and deterministic scorer."""

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        additional = str(record.get('additional') or '')
        question = render_question(str(record['question']), additional)
        images = record.get('image') or []

        image_indices = [int(index) for index in _IMAGE_PATTERN.findall(question)]
        expected_indices = list(range(len(images)))
        if image_indices != expected_indices:
            raise ValueError(
                f'VisFactor record {record.get("index")} references images {image_indices}, '
                f'but provides indices {expected_indices}.'
            )

        image_map = {
            index: self._normalize_media_value(image, media_type='image') for index, image in enumerate(images)
        }
        question = _IMAGE_PATTERN.sub(lambda match: f'<image_{match.group(1)}>', question)
        content: List[Content] = self._parse_text_with_images(text=question, image_map=image_map)

        return Sample(
            input=[ChatMessageUser(content=content)],
            target=str(record['answer']),
            metadata={
                'index': record['index'],
                'category_id': str(record['category_id']),
                'category_name': str(record['category_name']),
                'eval_index': int(record['eval_index']),
                'additional': additional,
            },
        )

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        metadata = task_state.metadata or {}
        normalized, correct = score_prediction(
            category_id=str(metadata['category_id']),
            prediction=original_prediction,
            reference=reference,
            additional=str(metadata.get('additional', '')),
        )
        return Score(
            value={'accuracy': correct},
            main_score_name='accuracy',
            extracted_prediction=normalized,
            prediction=original_prediction,
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        rows = []
        category_names = {}
        for sample_score in sample_scores:
            metadata = sample_score.sample_metadata or {}
            category_id = str(metadata['category_id'])
            category_names[category_id] = str(metadata['category_name'])
            rows.append((category_id, int(metadata['eval_index']), float(sample_score.score.main_value)))

        category_scores = aggregate_item_accuracy(rows)
        if not category_scores:
            return []

        total_items = sum(item_count for _, item_count in category_scores.values())
        overall = sum(accuracy for accuracy, _ in category_scores.values()) / len(category_scores)
        results = [
            AggScore(
                metric_name='accuracy',
                aggregation='macro_mean',
                dimensions={'scope': 'categories'},
                score=overall,
                num=total_items,
                metadata={'category_count': len(category_scores)},
            )
        ]
        results.extend(
            AggScore(
                metric_name='accuracy',
                aggregation='mean',
                dimensions={'category_id': category_id, 'category_name': category_names[category_id]},
                score=accuracy,
                num=item_count,
            )
            for category_id, (accuracy, item_count) in sorted(category_scores.items())
        )
        return results
