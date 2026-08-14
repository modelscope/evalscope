# GSM8K-Indic


## Overview

GSM8K-Indic translates the GSM8K grade-school math word problems into 10 Indic languages, each
available in native script and a romanized (Latin transliteration) variant, plus the original English.

## Task Description

- **Task Type**: Multilingual Mathematical Word Problem Solving
- **Input**: Grade-school math word problem in one of 21 language/script variants
- **Output**: Numerical answer derived through step-by-step reasoning
- **Languages**: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil,
  Telugu — each Indic language in both native script and a `_roman` transliterated variant

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split, the only split available)
- Use `subset_list` to evaluate specific languages/scripts (e.g., `['hi', 'hi_roman']`)
- Gold answers are the original English reasoning chain's final numeric value; only the question is
  translated


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `gsm8k_indic` |
| **Dataset ID** | [sarvamai/gsm8k-indic](https://modelscope.cn/datasets/sarvamai/gsm8k-indic/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `MultiLingual`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gsm8k_indic \
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
    datasets=['gsm8k_indic'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
