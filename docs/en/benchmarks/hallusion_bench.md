# HallusionBench


## Overview

HallusionBench is an advanced diagnostic benchmark designed to evaluate image-context reasoning and detect hallucination tendencies in Large Vision-Language Models (LVLMs). It specifically tests models' susceptibility to language hallucination and visual illusion.

## Task Description

- **Task Type**: Hallucination Detection and Visual Reasoning
- **Input**: Image + yes/no question about image content
- **Output**: YES or NO answer
- **Domains**: Hallucination detection, visual reasoning, factual accuracy

## Key Features

- Specifically designed to probe hallucination behaviors
- Tests both language hallucination and visual illusion
- Organized by categories and subcategories for detailed analysis
- Uses grouped accuracy metrics for robust evaluation
- Questions require precise image-context reasoning

## Evaluation Notes

- Default evaluation uses the **image** split
- Multiple accuracy metrics:
  - **aAcc**: Answer-level accuracy (per-question)
  - **fAcc**: Figure-level accuracy (all questions per figure correct)
  - **qAcc**: Question-level accuracy (grouped by question type)
- Requires simple YES/NO answers without explanation
- Aggregation at subcategory, category, and overall levels


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `hallusion_bench` |
| **Dataset ID** | [lmms-lab/HallusionBench](https://modelscope.cn/datasets/lmms-lab/HallusionBench/summary) |
| **Paper** | N/A |
| **Tags** | `Hallucination`, `MultiModal`, `Yes/No` |
| **Metrics** | `aAcc`, `qAcc`, `fAcc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `image` |
| **Aggregation** | `f1` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 951 |
| Prompt Length (Mean) | 136.78 chars |
| Prompt Length (Min/Max) | 76 / 292 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 951 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 388x56 - 5291x4536 |
| Formats | png |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "ba75d669",
      "content": [
        {
          "text": "Is China, Hongkong SAR, the leading importing country of gold, silverware, and jewelry with the highest import value in 2018?\nPlease answer YES or NO without an explanation."
        },
        {
          "image": "[BASE64_IMAGE: png, ~143.0KB]"
        }
      ]
    }
  ],
  "target": "NO",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "category": "VS",
    "subcategory": "chart",
    "visual_input": "1",
    "set_id": "0",
    "figure_id": "1",
    "question_id": "0",
    "gt_answer": "0",
    "gt_answer_details": "Switzerland is the leading importing country of gold, silverware, and jewelry with the highest import value in 2018?"
  }
}
```

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
    --datasets hallusion_bench \
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
    datasets=['hallusion_bench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


