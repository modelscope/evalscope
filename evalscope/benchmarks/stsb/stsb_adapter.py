from typing import Any

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters.decision_adapter import DecisionAdapter
from evalscope.api.dataset import Sample
from evalscope.api.metric import MetricSelector
from evalscope.api.registry import register_benchmark


@register_benchmark(
    BenchmarkMeta(
        name='stsb',
        pretty_name='STS-B Decision',
        dataset_id='sentence-transformers/stsb',
        eval_split='test',
        train_split=None,
        evaluation_version='v1.0',
        subset_list=['default'],
        metric_list=['mae'],
        primary_metric=MetricSelector(name='mae', aggregation='mean'),
        description="""
## Overview

Sentence similarity rated on six ordered levels from 0 to 5; continuous gold is preserved.

## Task Description

- **Task Type**: Typed text decisions
- **Input**: State and one question with named options or ordered levels
- **Output**: Choice, probability of yes, or ordinal score distribution
- **Domain**: Text understanding and decision making

## Key Features

- Uses the public test split of `sentence-transformers/stsb` on ModelScope
- Converts source similarity labels from [0, 1] to [0, 5], retaining fractional gold values
- Preserves option order, source IDs and rubric descriptions
- Supports System One and native AutoTrust APIs within their declared capabilities

## Evaluation Notes

- Zero-shot deterministic grading; gold answers and provenance never enter requests
- Primary: MAE of the expected Score value; also cross-sample Spearman and Pearson correlations
- Does not discretize continuous gold for accuracy or ranked probability scores
- Invalid responses are excluded from ordinary metrics and reflected in execution coverage
- Six levels (0-5) work with both protocols; constant or insufficient inputs leave correlations undefined
""",
    )
)
class StsbAdapter(DecisionAdapter):
    """Evaluate STS-B Decision through a typed decision service."""

    def record_to_sample(self, record: dict[str, Any]) -> Sample:
        """Keep the public normalized gold on its original continuous similarity scale."""
        return super().record_to_sample(
            {
                'state': {'sentence1': record['sentence1'], 'sentence2': record['sentence2']},
                'question': {
                    'type': 'score',
                    'instructions': 'Rate the semantic similarity of sentence1 and sentence2.',
                    'criteria': [
                        'Unrelated',
                        'Very different',
                        'Some shared meaning',
                        'Mostly similar',
                        'Nearly equivalent',
                        'Completely equivalent',
                    ],
                },
                'expected': float(record['score']) * 5,
                'continuous': True,
            }
        )
