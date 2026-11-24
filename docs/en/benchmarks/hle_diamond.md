# Humanity's-Last-Exam-Diamond


## Overview

Humanity's Last Exam Diamond (HLE-Diamond) is the official refined 1,000-question subset of Humanity's Last Exam. It retains the original expert-level question format while providing separate reasoning and knowledge partitions.

## Task Description

- **Task Type**: Expert-Level Question Answering
- **Input**: Question with an optional image
- **Output**: Answer with explanation and confidence score
- **Domain**: Broad academic and professional disciplines, with reasoning and knowledge partitions

## Key Features

- 1,000 questions: 500 reasoning and 500 knowledge questions
- Uses the original HLE answer types and multimodal questions
- Reports overall accuracy and separate reasoning/knowledge partition results
- Preserves the original question metadata, including subject category and rationale

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** with the same JSON C/I LLM judge contract as HLE
- Response format includes: Explanation, Answer, and Confidence (0-100%)
- Set `extra_params["include_multi_modal"]` to `False` for text-only models
- Hugging Face access requires accepting the dataset conditions and providing an access token; an authenticated or local dataset path is also supported


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `hle_diamond` |
| **Dataset ID** | [cais/hle-diamond](https://modelscope.cn/datasets/cais/hle-diamond/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `QA`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,000 |
| Prompt Length (Mean) | 890.94 chars |
| Prompt Length (Min/Max) | 232 / 18201 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `reasoning` | 500 | 1065.99 | 263 | 10668 |
| `knowledge` | 500 | 715.89 | 232 | 18201 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 272 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 329x12 - 14950x2780 |
| Formats | gif, jpeg, png, webp |


## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_multi_modal` | `bool` | `True` | Include multi-modal (image) questions during evaluation. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets hle_diamond \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
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
    datasets=['hle_diamond'],
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    dataset_args={
        'hle_diamond': {
            # subset_list: ['reasoning', 'knowledge']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
