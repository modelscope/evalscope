from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters.decision_adapter import DecisionAdapter
from evalscope.api.metric import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import PROBABILITY_SUM_TOLERANCE


@register_benchmark(
    BenchmarkMeta(
        name='open_decisions',
        pretty_name='OpenDecisions',
        tags=[Tags.DECISION],
        dataset_id='evalscope/OpenDecisions',
        eval_split='test',
        train_split=None,
        evaluation_version='v1.0',
        subset_list=['default'],
        metric_list=['capability'],
        primary_metric=MetricSelector(name='capability', aggregation='official'),
        description="""
## Overview

OpenDecisions: 2352 typed decisions. Loads the ModelScope test split for evaluation only.

## Task Description

- **Task Type**: Typed text decisions
- **Input**: State and one question with named options or ordered levels
- **Output**: Choice, probability of yes, or ordinal score distribution
- **Domain**: Text understanding and decision making

## Key Features

- 1178 Choice, 586 Noul and 588 Score tasks; multilingual inputs
- Dataset: [jgeuter/OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions); data CC BY-SA 4.0 with per-item provenance, scorer MIT
- Preserves option order, source IDs and rubric descriptions
- Supports System One and native AutoTrust APIs within their declared capabilities

## Evaluation Notes

- Zero-shot deterministic grading; gold answers and provenance never enter requests
- Primary: Capability; official Intelligence and Calibration use the pinned v1.0 scorer at revision `5026a13ad6e5a8606810a3a5c4e410d8d6638e71`
- Official values are stored as ratios (displayed as percentages); missing predictions count as wrong in Intelligence
- Ordinary probability metrics use valid predictions only. Noul threshold is 0.5 in the official protocol
- Invalid responses are excluded from ordinary metrics and reflected in execution coverage
- Score rubrics have 3-6 levels. AutoTrust fixed-six-level services cannot evaluate the complete dataset
- Limit/shuffle runs score the selected items and must not be compared with a complete leaderboard run
""",
    )
)
class OpenDecisionsAdapter(DecisionAdapter):
    """Evaluate OpenDecisions through a typed decision service."""

    aggregate_empty_scores = True
    probability_sum_tolerance = PROBABILITY_SUM_TOLERANCE
    normalization_threshold = PROBABILITY_SUM_TOLERANCE

    def aggregate_subset(self, subset: str, sample_scores: list) -> list:
        """Score every requested task in this subset, including missing predictions."""
        from .utils import aggregate_official

        tasks = [
            {
                'id': sample.id,
                'question': {'type': sample.metadata['decision_kind']},
                'labels': sample.metadata['decision_labels'],
                'expected': sample.metadata['decision_gold'],
                'tier': sample.metadata['tier'],
            }
            for sample in self.test_dataset[subset]
        ]
        return super().aggregate_scores(sample_scores) + aggregate_official(sample_scores, tasks)
