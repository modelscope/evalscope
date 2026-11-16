# Copyright (c) Alibaba, Inc. and its affiliates.
from statistics import harmonic_mean
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.metric.semantics import MetricIdentity
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.metrics.semantics import get_semantics_resolver
from evalscope.report import Category, Metric, Report, Subset

from .utils import extract_answer, match_answer

# The 23 BBEH tasks; the dataset's `task` column uses the same names with spaces instead of underscores.
SUBSET_LIST = [
    'boardgame_qa',
    'boolean_expressions',
    'buggy_tables',
    'causal_understanding',
    'disambiguation_qa',
    'dyck_languages',
    'geometric_shapes',
    'hyperbaton',
    'linguini',
    'movie_recommendation',
    'multistep_arithmetic',
    'nycc',
    'object_counting',
    'object_properties',
    'sarc_triples',
    'shuffled_objects',
    'spatial_reasoning',
    'sportqa',
    'temporal_sequence',
    'time_arithmetic',
    'web_of_lies',
    'word_sorting',
    'zebra_puzzles',
]

# Suffix the BBEH paper (arXiv:2502.19187, Appendix) adds to every question.
PROMPT_TEMPLATE = (
    '{question}\n\n'
    'Think step by step, and when you provide the final answer, please use the prefix "The answer is:" without any '
    'modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: '
    '"The answer is: 42" or "The answer is: yes". If the question is multiple choice with a single correct answer, '
    'the final answer must only be the letter corresponding to the correct answer. For example, "The answer is: (a)".'
)

DESCRIPTION = """
## Overview

BIG-Bench Extra Hard (BBEH) evaluates challenging general reasoning across 23 tasks. It builds on BIG-Bench Hard with harder problems covering a similar range of reasoning skills.

## Task Description

- **Task Type**: General reasoning (free-form and multiple-choice answers)
- **Input**: A reasoning question with answer-format instructions
- **Output**: Reasoning followed by `The answer is: <answer>`
- **Domain**: Logic, arithmetic, language, commonsense, temporal and spatial reasoning, humour and sarcasm

## Key Features

- Full benchmark: 4,520 examples across 23 tasks (200 per task, except 120 for Disambiguation QA)
- Optional BBEH Mini: 460 examples, with 20 per task, for quicker evaluation
- Varied question formats and long inputs, with prompts reaching about 32,500 characters

## Evaluation Notes

- Uses the official zero-shot prompt and answer matching rules, including their punctuation-related limitations.
- Reports per-task accuracy, overall micro-averaged accuracy, and the paper's BBEH Score.
- BBEH Score is the harmonic mean of task accuracies after adding one percentage point to each. Its range is 1% to 101%, rather than the usual 0% to 100% accuracy range.
- Evaluate all 23 tasks for overall comparisons. Allow enough output tokens: responses without a final answer may be marked incorrect.
"""


@register_benchmark(
    BenchmarkMeta(
        name='bbeh',
        pretty_name='BBEH',
        dataset_id='evalscope/bbeh',
        hf_dataset_id='BBEH/bbeh',
        paper_url='https://arxiv.org/abs/2502.19187',
        tags=[Tags.REASONING],
        description=DESCRIPTION.strip(),
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # The dataset only provides a 'train' split
        metric_list=['acc'],
        prompt_template=PROMPT_TEMPLATE,
        evaluation_version='v1.0',
        extra_params={
            'mini': {
                'type': 'bool',
                'description': 'Evaluate BBEH Mini, the official 460-example subset (20 per task).',
                'value': False,
            }
        },
    )
)
class BBEHAdapter(DefaultDataAdapter):
    """Score BBEH with the official answer extraction and fuzzy match."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True
        self.mini = bool(self.extra_params.get('mini', False))

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['input'],
            target=record['target'],
            subset_key=record['task'].replace(' ', '_'),
            metadata={'mini': bool(record['mini'])},
        )

    def sample_filter(self, sample: Sample) -> bool:
        return not self.mini or sample.metadata['mini']

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return extract_answer(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        return Score(
            value={'acc': float(match_answer(filtered_prediction, reference))},
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

    def _on_generate_report_end(self, report: Report, output_dir: str, **kwargs: Any) -> None:
        """Append the BBEH headline metric: the adjusted harmonic mean of per-task accuracies."""
        accuracy = next((metric for metric in report.metrics if metric.identity.name == 'accuracy'), None)
        if accuracy is None:
            return
        subsets = [subset for category in accuracy.categories for subset in category.subsets if not subset.is_aggregate]
        if not subsets:
            return
        # Recover exact accuracies from counts; report scores are already rounded.
        accuracies = (
            subset.num_correct / subset.num if subset.num_correct is not None and subset.num else subset.score
            for subset in subsets
        )
        score = harmonic_mean(value + 0.01 for value in accuracies)
        identity = MetricIdentity(name='bbeh_score', aggregation='harmonic_mean')
        report.metrics.append(
            Metric(
                identity=identity,
                categories=[Category(name='-', subsets=[Subset(name='OVERALL', score=score, num=accuracy.num)])],
                semantics=get_semantics_resolver().resolve(self.name, identity).semantics,
            )
        )
