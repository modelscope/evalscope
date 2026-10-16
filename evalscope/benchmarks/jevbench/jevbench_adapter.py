import json
from typing import Any

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters.decision_adapter import DecisionAdapter
from evalscope.api.dataset import Sample
from evalscope.api.metric import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='jevbench',
        pretty_name='JevBench public subset',
        tags=[Tags.DECISION],
        dataset_id='evalscope/jevbench',
        eval_split='test',
        train_split=None,
        evaluation_version='v1.0',
        subset_list=['default'],
        metric_list=['public_accuracy'],
        primary_metric=MetricSelector(name='public_accuracy', aggregation='official'),
        description="""
## Overview

Public 231 Benchmark Heaven decisions, preserving original rubrics and labels.

## Task Description

- **Task Type**: Typed text decisions
- **Input**: State and one question with named options or ordered levels
- **Output**: Choice, probability of yes, or ordinal score distribution
- **Domain**: Text understanding and decision making

## Key Features

- Pinned Benchmark Heaven public snapshot: 72 original, 48 easy and 111 hard tasks (139 Choice, 74 Noul, 18 Score)
- Source: [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench/tree/609a664e81f46e03731600bd7bd9f1b7a61257c5); MIT-licensed public records
- Preserves option order, source IDs and rubric descriptions
- Uses the ModelScope public-task mirror; the official Hugging Face JevBench repository contains leaderboard results, not these task records
- Supports System One and native AutoTrust APIs within their declared capabilities

## Evaluation Notes

- Zero-shot deterministic grading; gold answers and provenance never enter requests
- Primary: public-subset accuracy (missing predictions count as wrong); also family-macro accuracy, per-kind probability metrics and Score expected-value errors
- Four- and five-level Score tasks keep their original rubrics; AutoTrust fixed-six-level services reject these tasks explicitly
- Invalid responses are excluded from ordinary metrics and reflected in execution coverage
- Public scorer: probability sum tolerance 0.001; normalize only above that within 0.02; Noul ties select no
- This public snapshot does not produce the full JevBench Composite or Capability leaderboard metrics
""",
    )
)
class JevbenchAdapter(DecisionAdapter):
    """Evaluate JevBench public subset through a typed decision service."""

    probability_sum_tolerance = 0.02
    aggregate_empty_scores = True
    normalization_threshold = 0.001
    noul_use_argmax = True

    def record_to_sample(self, record: dict[str, Any]) -> Sample:
        """Unpack the lossless JSON column used by the public Parquet mirror."""
        if 'record_json' in record:
            record = json.loads(record['record_json'])
        return super().record_to_sample(record)

    def aggregate_subset(self, subset: str, sample_scores: list) -> list:
        """Include missing public predictions in the accuracy and family-macro denominators."""
        from collections import defaultdict

        from evalscope.api.metric import AggScore

        aggregates = [a for a in super().aggregate_scores(sample_scores) if a.metric_name != 'family_macro']
        predictions = {
            sample.sample_id: sample.score.metadata['decision']['correct']
            for sample in sample_scores
            if (sample.score.metadata or {}).get('decision')
        }
        families = defaultdict(list)
        outcomes = []
        for sample in self.test_dataset[subset]:
            correct = predictions.get(sample.id, 0.0)
            families[sample.metadata['family']].append(correct)
            outcomes.append(correct)
        if outcomes:
            metadata = {
                'missing_predictions_count_as_wrong': True,
                'valid_predictions': len(predictions),
                'strict_sum_tolerance': 0.001,
                'headline_sum_tolerance': 0.02,
            }
            aggregates += [
                AggScore(
                    metric_name='public_accuracy',
                    aggregation='official',
                    score=sum(outcomes) / len(outcomes),
                    num=len(outcomes),
                    metadata=metadata,
                ),
                AggScore(
                    metric_name='family_macro',
                    aggregation='official',
                    score=sum(sum(v) / len(v) for v in families.values()) / len(families),
                    num=len(outcomes),
                    metadata={**metadata, 'families': len(families)},
                ),
            ]
        return aggregates
