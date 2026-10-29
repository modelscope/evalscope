# Copyright (c) Alibaba, Inc. and its affiliates.

import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from .utils import build_task_prompt, parse_grid_from_response

logger = get_logger()

DESCRIPTION = """
## Overview

ARC-AGI-2 (Abstraction and Reasoning Corpus for Artificial General Intelligence 2) is a benchmark designed to measure an AI system's ability to efficiently acquire new skills on-the-fly, using only a handful of demonstrations. It evaluates abstract reasoning and pattern recognition through grid transformation tasks.

## Task Description

- **Task Type**: Abstract Reasoning / Pattern Recognition
- **Input**: A series of input-output grid pairs (demonstrations) followed by a test input grid
- **Output**: The predicted output grid matching the inferred transformation rule
- **Grid Format**: 2D arrays of integers (0-9), variable sizes (up to 30x30)

## Key Features

- 1,000 public training tasks and 120 public evaluation tasks (173 test inputs)
- Each task provides 2-10 demonstration input/output pairs
- Models must infer the transformation rule from demonstrations
- Tests abstract reasoning without reliance on learned knowledge
- Pixel-perfect output required (exact grid match)

## Evaluation Notes

- Scoring is based on **exact grid match** (shape and all values must be identical)
- A task with several test inputs is asked once per test input; as in the official ARC Prize scorer, a task scores
  the fraction of its test inputs solved, and tasks are averaged
- Models must output the grid as a JSON 2D array
- Zero-shot evaluation (demonstrations are provided within each task)
- Designed to be solvable by humans but challenging for AI
"""

SYSTEM_PROMPT = (
    'You are an expert at abstract reasoning and pattern recognition. '
    'Given input-output grid pairs as examples, you must figure out the transformation rule '
    'and apply it to a new test input to produce the correct output grid.'
)

PROMPT_TEMPLATE = '{question}'


@register_benchmark(
    BenchmarkMeta(
        name='arc_agi_2',
        pretty_name='ARC-AGI-2',
        dataset_id='evalscope/arc-agi-2',
        hf_dataset_id='arc-agi-community/arc-agi-2',
        tags=[Tags.REASONING],
        description=DESCRIPTION,
        subset_list=['default'],
        metric_list=['acc'],
        aggregation='mean_and_pass_hat_k',
        primary_metric=MetricSelector(name='accuracy', aggregation='pass_hat_k', dimensions={'k': 1}),
        few_shot_num=0,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        system_prompt=SYSTEM_PROMPT,
        evaluation_version='v1.1',
    )
)
class ArcAgi2Adapter(DefaultDataAdapter):
    def record_to_sample(self, record: Dict[str, Any]) -> List[Sample]:
        # One sample per test input; the task id groups them back for scoring.
        task_id = hashlib.sha1(json.dumps(record['question']).encode()).hexdigest()[:16]
        return [
            Sample(
                input=build_task_prompt(record, pair_index),
                target=json.dumps(pair['output']),
                metadata={'task_id': task_id, 'pair_index': pair_index},
            )
            for pair_index, pair in enumerate(record['question'])
        ]

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract a grid from model prediction and return as JSON string."""
        grid = parse_grid_from_response(prediction)
        if grid:
            return json.dumps(grid)
        return prediction.strip()

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Exact grid match scoring."""
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        try:
            pred_grid = json.loads(filtered_prediction)
            ref_grid = json.loads(reference)
            correct = pred_grid == ref_grid
        except (json.JSONDecodeError, TypeError):
            correct = False
        score.value = {'acc': 1.0 if correct else 0.0}
        score.main_score_name = 'acc'
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Average the test inputs of each task, then the tasks, as the ARC Prize scorer does."""
        tasks = defaultdict(list)
        for sample_score in sample_scores:
            tasks[(sample_score.sample_metadata or {}).get('task_id', sample_score.sample_id)].append(sample_score)
        merged = defaultdict(list)
        for task_scores in tasks.values():
            for agg in super().aggregate_scores(task_scores):
                merged[(agg.metric_name, agg.aggregation, json.dumps(agg.dimensions, sort_keys=True))].append(agg)
        return [
            AggScore(
                score=sum(agg.score for agg in group) / len(group),
                metric_name=group[0].metric_name,
                aggregation=group[0].aggregation,
                dimensions=group[0].dimensions,
                num=len(group),
                ids=[sample_id for agg in group for sample_id in agg.ids or []],
            )
            for group in merged.values()
        ]
