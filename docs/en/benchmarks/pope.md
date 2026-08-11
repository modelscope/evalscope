# POPE


## Overview

POPE (Polling-based Object Probing Evaluation) is a benchmark specifically designed to evaluate object hallucination in Large Vision-Language Models (LVLMs). It tests models' ability to accurately identify objects present in images through yes/no questions.

## Task Description

- **Task Type**: Object Hallucination Detection (Yes/No Q&A)
- **Input**: Image with question "Is there a [object] in the image?"
- **Output**: YES or NO answer
- **Focus**: Measuring accuracy vs. hallucination rate

## Key Features

- Three sampling strategies: random, popular, adversarial
- Tests for false positive object claims (hallucination)
- Based on MSCOCO images
- Simple yes/no question format for objective evaluation
- Measures alignment between model responses and visual content

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Five metrics: accuracy, precision, recall, F1 score, yes_ratio
- Accuracy is the primary metric; precision, recall, F1, and yes_ratio provide supporting diagnostics
- Three subsets: `popular`, `adversarial`, `random`
- "Popular" and "adversarial" subsets are more challenging
- yes_ratio indicates model's tendency to answer "yes"


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `pope` |
| **Dataset ID** | [lmms-lab/POPE](https://modelscope.cn/datasets/lmms-lab/POPE/summary) |
| **Paper** | N/A |
| **Tags** | `Hallucination`, `MultiModal`, `Yes/No` |
| **Metrics** | `accuracy`, `precision`, `recall`, `f1`, `yes_ratio` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `N/A` |
| **Aggregation** | `f1` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
{question}
Please answer YES or NO without an explanation.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets pope \
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
    datasets=['pope'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
