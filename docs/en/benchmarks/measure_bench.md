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
- Primary metric: **Accuracy** (`accuracy`) — ``all_correct``: number *and* unit both correct
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
| **Metrics** | `accuracy`, `number_acc`, `unit_acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `real_world` |


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
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
