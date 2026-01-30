# ZeroBench


## Overview

ZeroBench is a challenging visual reasoning benchmark for Large Multimodal Models (LMMs). It consists of 100 high-quality, manually curated questions covering numerous domains, reasoning types, and image types designed to be beyond current model capabilities.

## Task Description

- **Task Type**: Advanced Visual Reasoning
- **Input**: One or more images + challenging visual reasoning question
- **Output**: Step-by-step reasoning with final answer in curly braces
- **Domains**: Visual reasoning, perception, multi-step inference

## Key Features

- 100 manually curated high-quality questions
- Designed to challenge frontier models (zero pass@1 with greedy decoding)
- Covers diverse domains, reasoning types, and image types
- No model achieves 5/5 reliability score
- Tests limits of current visual reasoning capabilities

## Evaluation Notes

- Default evaluation uses the **zerobench** split
- Primary metric: **Accuracy** with LLM judge
- Answers must be in format: `{final answer}`
- Includes subquestions split for detailed analysis
- Uses image compression to handle large images


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `zerobench` |
| **Dataset ID** | [evalscope/zerobench](https://modelscope.cn/datasets/evalscope/zerobench/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `zerobench` |
| **Train Split** | `zerobench_subquestions` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 100 |
| Prompt Length (Mean) | 645.72 chars |
| Prompt Length (Min/Max) | 139 / 1998 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 108 |
| Images per Sample | min: 1, max: 3, mean: 1.08 |
| Resolution Range | 512x297 - 5559x4070 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "f3276b25",
      "content": [
        {
          "text": "I want to purchase all the Montellier bottles from the top three shelves. How much do I save by purchasing the bottles with a loyalty card? Give your final answer in dollars.\n\n\n\nLet's think step by step and give the final answer in curly braces,\nlike this: {final answer}\"\n"
        },
        {
          "image": "[BASE64_IMAGE: png, ~462.4KB]"
        }
      ]
    }
  ],
  "target": "11.90",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "1",
    "question_images": [
      "images/1_0.png"
    ],
    "image_attribution": "Own"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}



Let's think step by step and give the final answer in curly braces,
like this: {{final answer}}"

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets zerobench \
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
    datasets=['zerobench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


