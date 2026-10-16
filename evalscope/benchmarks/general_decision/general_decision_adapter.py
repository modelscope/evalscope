from typing import Any

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters.decision_adapter import DecisionAdapter
from evalscope.api.dataset import DatasetDict
from evalscope.api.metric import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='general_decision',
        pretty_name='General Decision',
        tags=[Tags.DECISION, Tags.CUSTOM],
        dataset_id='general_decision',
        eval_split='test',
        train_split=None,
        evaluation_version='v1.1',
        subset_list=['default'],
        metric_list=['acc'],
        primary_metric=MetricSelector(name='accuracy', aggregation='mean'),
        description="""
## Overview

Evaluate local JSONL datasets containing one Choice, Noul or Score question per record.

## Task Description

- **Task Type**: Text decisions: Choice, Noul and Score
- **Input**: State and one question with named options or ordered levels
- **Output**: Choice, probability of yes, or ordinal score distribution
- **Domain**: Text understanding and decision making

## Key Features

- Reads `default_test.jsonl` from a directory passed through `dataset_args.general_decision.local_path`
- Fields: `id`, `state`, `question` (type/instructions/criteria), `expected`; one question per record
- Preserves option order, source IDs and rubric descriptions
- Supports System One and native AutoTrust APIs within their declared capabilities

## Evaluation Notes

- Rule-based zero-shot grading; reference answers and provenance are used only for scoring, never as request fields
- Default primary metric: accuracy on discrete reference labels; continuous Score runs can select MAE explicitly
- Probability metrics are separated by type; Score retains its expectation and normalized RPS
- Invalid responses are excluded from ordinary metrics and reflected in execution coverage
- This is a local data entry point; no upload is required
- See [Custom decision datasets](../advanced_guides/custom_dataset/decision.md) for runnable JSONL examples,
  required and optional fields, question types, metrics, continuous scores, and configuration
- See the [decision evaluation best practice](../best_practice/decision_evaluation.md) for deployment and Web UI
""",
    )
)
class GeneralDecisionAdapter(DecisionAdapter):
    """Evaluate General Decision through a typed decision service."""

    requires_local_dataset = True

    def load_from_disk(self, **kwargs: Any) -> tuple[DatasetDict, DatasetDict | None]:
        """Read named local files directly, preserving heterogeneous decision records."""
        return super().load_from_disk(use_local_loader=True)
