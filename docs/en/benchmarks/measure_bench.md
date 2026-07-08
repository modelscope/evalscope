# MeasureBench


## Overview

MeasureBench is a comprehensive benchmark for evaluating the ability of vision-language models (VLMs) to read values from measuring instruments. It covers both **real-world photographs** and **synthetically generated images** of 26 instrument types across 4 design categories.

## Task Description

- **Task Type**: Free-form Visual Question Answering (instrument reading)
- **Input**: An image of a measuring instrument + a reading question
- **Output**: The instrument's current reading (numeric value or time, with unit)
- **Domains**: Ammeters, clocks, thermometers, scales, speedometers, and 21 more instrument types

## Key Features

- 2,442 total samples across two splits: real_world (1,272) and synthetic_test (1,170)
- 26 instrument types, 4 design categories (dial, digital, analog, linear)
- Accepts a tolerance interval around the correct value rather than requiring an exact match
- For clocks: handles both 12-hour and 24-hour ambiguity via multiple valid intervals
- Unit recognition is evaluated separately from numeric accuracy

## Evaluation Notes

- Default splits: **real_world** and **synthetic_test** (treated as separate subsets)
- Primary metric: **Accuracy** (acc) — ``all_correct``: number *and* unit both correct
- Secondary metrics: **number_acc** (numeric only), **unit_acc** (unit only)
- Two evaluators: ``interval_matching`` (single valid range) and ``multi_interval_matching`` (e.g. clock AM/PM)
- Model output is expected in the format ``Answer: <value> <unit>`` on the last line
- ``image_type`` is recorded in each sample's metadata; per-type results are visible in the
  ``subset_key`` column of review files but are not separately selectable via ``subset_list``
- [Paper](https://arxiv.org/abs/2510.26865) | [GitHub](https://github.com/flageval-baai/MeasureBench)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `measure_bench` |
| **Dataset ID** | [evalscope/MeasureBench](https://modelscope.cn/datasets/evalscope/MeasureBench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2510.26865) |
| **Tags** | `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `acc`, `number_acc`, `unit_acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `real_world` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,442 |
| Prompt Length (Mean) | 150.9 chars |
| Prompt Length (Min/Max) | 126 / 215 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `real_world` | 1,272 | 153.83 | 131 | 215 |
| `synthetic_test` | 1,170 | 147.71 | 126 | 192 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,442 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 108x79 - 3025x1599 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `real_world`

```json
{
  "input": [
    {
      "id": "1341f508",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~75.8KB]"
        },
        {
          "text": "What is the reading of the instrument?\nProvide your final answer on the last line in the format: Answer: <value> <unit>. For example: Answer: 42.5 A"
        }
      ]
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "ammeter",
  "metadata": {
    "question_id": "ammeter_0",
    "image_type": "ammeter",
    "design": "dial",
    "evaluator": "interval_matching",
    "evaluator_kwargs": "{\"interval\": [9.5, 9.7], \"units\": [\"A\", \"Ampere\"]}"
  }
}
```

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets measure_bench \
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
    datasets=['measure_bench'],
    dataset_args={
        'measure_bench': {
            # subset_list: ['real_world', 'synthetic_test']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
