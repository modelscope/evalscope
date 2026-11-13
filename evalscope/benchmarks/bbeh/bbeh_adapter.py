# Copyright (c) Alibaba, Inc. and its affiliates.
from statistics import harmonic_mean
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.report import Report

from .utils import fuzzy_match, preprocess_reference, preprocess_sample

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

BIG-Bench Extra Hard (BBEH) replaces each of the 23 tasks in BIG-Bench Hard with a new task that probes a similar reasoning skill but is much harder. Frontier models that saturate BBH score far lower on BBEH, so it separates general reasoning ability across models again.

## Task Description

- **Task Type**: General reasoning (free-form and multiple-choice answers)
- **Input**: A task-specific question followed by the official answer-format instruction
- **Output**: Step-by-step reasoning ending in `The answer is: <answer>`
- **Domain**: Logical, temporal, spatial, causal, arithmetic, linguistic and commonsense reasoning, humour and sarcasm understanding

## Key Features

- 4,520 examples over 23 tasks (200 per task, 120 for Disambiguation QA)
- BBEH Mini: the official 460-example subset (20 per task), selected with `extra_params={'mini': True}`
- Answers are extracted and matched with the official `evaluate.py` from google-deepmind/bbeh
- Long inputs: prompts average about 4,300 characters and reach 32,000 (Shuffled Objects and Zebra Puzzles average over 11,000)

## Evaluation Notes

- 0-shot, with the prompt suffix used in the BBEH paper.
- `acc` per task uses the official fuzzy match (case-insensitive, `(a)` vs `a`, numeric equality, LaTeX `\\boxed{}` and markup removal).
- The overall `acc` is the micro average over all examples, which the BBEH leaderboard reports as "Micro Avg".
- The report adds the paper's headline metric, `accuracy` with `harmonic_mean` aggregation: the harmonic mean of per-task accuracies, with 1 percentage point added to each to handle zero scores, as in the paper.
- Give models a generous `max_tokens`; the paper attributes below-random scores to answers cut off before the final line.
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
        return preprocess_sample(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        # `filtered_prediction` is already `preprocess_sample` of the response (see `extract_answer`).
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        score.value = {'acc': float(fuzzy_match(filtered_prediction, preprocess_reference(reference)))}
        return score

    def _on_generate_report_end(self, report: Report, output_dir: str, **kwargs: Any) -> None:
        """Append the BBEH headline metric: the adjusted harmonic mean of per-task accuracies."""
        from evalscope.api.metric.semantics import MetricIdentity
        from evalscope.metrics.semantics import get_semantics_resolver
        from evalscope.report import Category, Metric, Subset

        accuracy = next((metric for metric in report.metrics if metric.identity.name == 'accuracy'), None)
        if accuracy is None:
            return
        subsets: List[Subset] = [
            subset for category in accuracy.categories for subset in category.subsets if not subset.is_aggregate
        ]
        if not subsets:
            return
        # The paper adds 1 to every accuracy (in percent) so that a single zero does not zero the mean.
        score = harmonic_mean([100 * subset.score + 1 for subset in subsets]) / 100
        identity = MetricIdentity(name='accuracy', aggregation='harmonic_mean')
        semantics = get_semantics_resolver().resolve(self.name, identity).semantics
        overall = Subset(name='OVERALL', score=score, num=sum(subset.num for subset in subsets))
        report.metrics.append(
            Metric(identity=identity, categories=[Category(name='-', subsets=[overall])], semantics=semantics)
        )
