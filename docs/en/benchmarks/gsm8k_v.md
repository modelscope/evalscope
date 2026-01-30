# GSM8K-V


## Overview

GSM8K-V is a purely visual multi-image mathematical reasoning benchmark that systematically transforms each GSM8K math word problem into its visual counterpart. It enables clean within-item comparison across modalities for multimodal math evaluation.

## Task Description

- **Task Type**: Visual Mathematical Word Problem Solving
- **Input**: Multiple images representing a math problem + question
- **Output**: Numerical answer in \boxed{} format
- **Domains**: Visual math reasoning, scene understanding, arithmetic

## Key Features

- Visual transformation of GSM8K text problems
- Multi-image input format for complete problem representation
- Enables text-vs-visual modality comparisons
- Categorized by problem type and subcategory
- Preserves original problem difficulty levels

## Evaluation Notes

- Default evaluation uses the **train** split
- Primary metric: **Accuracy** with numeric comparison
- Answers should be in \boxed{} format with number only
- Step-by-step reasoning encouraged before final answer
- Metadata includes original text question for comparison


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `gsm8k_v` |
| **Dataset ID** | [evalscope/GSM8K-V](https://modelscope.cn/datasets/evalscope/GSM8K-V/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `MultiModal`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,319 |
| Prompt Length (Mean) | 492.4 chars |
| Prompt Length (Min/Max) | 410 / 789 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 5,344 |
| Images per Sample | min: 2, max: 11, mean: 4.05 |
| Resolution Range | 1024x1024 - 1024x1024 |
| Formats | jpeg |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "9e0072fd",
      "content": [
        {
          "text": "You are an expert at solving mathematical word problems. Please solve the following problem step by step, showing your reasoning.\n\nWhen providing your final answer:\n- If the answer can be expressed as a whole number (integer), provide it as a ... [TRUNCATED] ... s Janet (white woman, shoulder-length brown hair, wearing a light blue blouse and khaki pants) make each day at the farmers' market?\n\nPlease think step by step. After your reasoning, put your final answer within \\boxed{} with the number only."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~153.1KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~175.1KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~124.7KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~172.8KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~117.4KB]"
        }
      ]
    }
  ],
  "target": "18.0",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 1,
    "category": "other",
    "subcategory": "count",
    "original_question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
You are an expert at solving mathematical word problems. Please solve the following problem step by step, showing your reasoning.

When providing your final answer:
- If the answer can be expressed as a whole number (integer), provide it as an integer
Problem: {question}

Please think step by step. After your reasoning, put your final answer within \boxed{{}} with the number only.

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gsm8k_v \
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
    datasets=['gsm8k_v'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


