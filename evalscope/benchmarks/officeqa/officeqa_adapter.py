# Copyright (c) Alibaba, Inc. and its affiliates.
# Scoring logic adapted from https://github.com/databricks/officeqa/blob/main/reward.py

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from .utils import DEFAULT_TOLERANCE, extract_final_answer, score_answer

logger = get_logger()

DESCRIPTION = """
## Overview

OfficeQA is a grounded reasoning benchmark by Databricks, built for evaluating model/agent performance on end-to-end grounded reasoning tasks over U.S. Treasury Bulletin documents (1939-2025).

## Task Description

- **Task Type**: Document-based Question Answering
- **Input**: A question requiring information from U.S. Treasury Bulletins
- **Output**: A precise answer (numeric values, text, or structured data)
- **Difficulty**: All questions in the Pro subset are rated "hard"

## Key Features

- 133 questions in the Pro subset (default) sourced from real U.S. Treasury bulletins
- Answers include formatted numbers (e.g., "2,602"), lists, text, and dates
- Tests precise factual extraction from official documentation
- Scoring uses fuzzy numeric matching with configurable tolerance

## Evaluation Notes

- Uses **rule-based scoring** adapted from official reward.py
- Numerical answers matched with configurable relative error tolerance (default 1%)
- Text answers use case-insensitive substring matching
- Supports multi-number list answers and unit-aware comparison
"""

PROMPT_TEMPLATE = '{question}\nPlease provide a precise and concise answer.'.lstrip()


@register_benchmark(
    BenchmarkMeta(
        name='officeqa',
        pretty_name='OfficeQA',
        dataset_id='evalscope/officeqa',
        tags=[Tags.QA, Tags.KNOWLEDGE],
        description=DESCRIPTION,
        subset_list=['default'],
        metric_list=['acc'],
        few_shot_num=0,
        eval_split='train',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class OfficeQAAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['question'],
            target=str(record['answer']),
            metadata={
                'uid': record['uid'],
                'source_files': record['source_files'],
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract final answer from prediction (handles <FINAL_ANSWER> tags)."""
        return extract_final_answer(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score using official OfficeQA fuzzy numeric matching."""
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        correct = score_answer(reference, filtered_prediction, tolerance=DEFAULT_TOLERANCE)
        score.value = {'acc': correct}
        score.main_score_name = 'acc'
        return score
