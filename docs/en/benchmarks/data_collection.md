# Data-Collection


## Overview

Data-Collection is a flexible framework for mixing multiple evaluation datasets into a unified evaluation suite. It enables comprehensive model assessment using carefully selected samples from various benchmarks.

## Task Description

- **Task Type**: Multi-Dataset Unified Evaluation
- **Input**: Mixed samples from multiple benchmark datasets
- **Output**: Aggregated scores across tasks, datasets, and categories
- **Flexibility**: Supports custom dataset collections

## Key Features

- Mix multiple benchmarks into one evaluation
- Hierarchical reporting (subset, dataset, task, tag, category levels)
- Sample-level weighting support
- Automatic adapter initialization for each dataset
- Comprehensive aggregation (micro, macro, weighted averages)

## Evaluation Notes

- Dataset must be pre-compiled as a collection
- Supports various task types (MCQ, QA, coding, etc.)
- Generates multi-level reports for detailed analysis
- See [Collection Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/collection/index.html) for usage


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `data_collection` |
| **Dataset ID** | N/A |
| **Paper** | N/A |
| **Tags** | `Custom` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets data_collection \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['data_collection'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


