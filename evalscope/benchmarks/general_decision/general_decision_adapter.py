from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters.decision_adapter import DecisionAdapter
from evalscope.api.metric import MetricSelector
from evalscope.api.registry import register_benchmark


@register_benchmark(
    BenchmarkMeta(
        name='general_decision',
        pretty_name='General Decision',
        dataset_id='general_decision',
        eval_split='test',
        train_split=None,
        evaluation_version='v1.0',
        subset_list=['default'],
        metric_list=['acc'],
        primary_metric=MetricSelector(name='accuracy', aggregation='mean'),
        description="""
## Overview

User-provided single-question typed decisions in local JSONL files.

## Task Description

- **Task Type**: Typed text decisions
- **Input**: State and one question with named options or ordered levels
- **Output**: Choice, probability of yes, or ordinal score distribution
- **Domain**: Text understanding and decision making

## Key Features

- Reads `default_test.jsonl` from a directory passed through `dataset_args.general_decision.local_path`
- Fields: `id`, `state`, `question` (type/instructions/criteria), `expected`; one question per record
- Preserves option order, source IDs and rubric descriptions
- Supports System One and native AutoTrust APIs within their declared capabilities

## Evaluation Notes

- Zero-shot deterministic grading; gold answers and provenance never enter requests
- Primary: accuracy; Choice/Noul use categorical gold, Score uses an integer level
- Probability metrics are separated by type; Score retains its expectation and normalized RPS
- Invalid responses are excluded from ordinary metrics and reflected in execution coverage
- This is a local data entry point; no upload is required
- See the decision evaluation best practice for examples and deployment limits
""",
    )
)
class GeneralDecisionAdapter(DecisionAdapter):
    """Evaluate General Decision through a typed decision service."""

    requires_local_dataset = True
