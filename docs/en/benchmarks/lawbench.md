# LawBench


## Overview

LawBench evaluates Chinese legal knowledge and reasoning across memorization, understanding and application. This integration supports 19 tasks, including legal exams, case analysis, information extraction and legal consultation. See the [official project](https://github.com/open-compass/LawBench) for the benchmark and paper.

## Task Description

- **Task Type**: Legal question answering, classification, extraction and generation
- **Input**: A Chinese legal question or document with task instructions
- **Output**: A Chinese response in the requested format
- **Domain**: Chinese legal knowledge and legal reasoning

## Key Features

- 9,500 supported samples, with 500 per task; select tasks using subsets `1-1`–`1-2`, `2-2`–`2-10`, and `3-1`–`3-8`
- Task groups cover legal knowledge, document understanding and practical application
- Data downloads automatically from [ModelScope](https://modelscope.cn/datasets/evalscope/LawBench)
- Uses the official task-specific scoring rules; no LLM judge is required

## Evaluation Notes

- Supports zero-shot evaluation. Set `subset_list` to evaluate selected tasks; use `limit` for a short trial.
- Document proofreading (task `2-1`) is not supported. The overall score covers the 19 supported tasks and is not the complete 20-task LawBench score.
- Install scoring dependencies with `pip install 'evalscope[lawbench]'`.
- `lawbench_score` follows each task's official metric. Higher is better; most scores are in [0, 1], while imprisonment prediction scores can be negative. Overall and category scores give equal weight to each evaluated task.
- `abstention_rate` reports how often a task's answer parser finds no valid answer. Officially excluded reference answers do not contribute to the score.
- This benchmark is distinct from PLawBench and does not require OpenCompass.
- For offline evaluation, set `local_path` to a downloaded copy of the ModelScope dataset, including its `README.md` subset configuration and data files.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `lawbench` |
| **Dataset ID** | [evalscope/LawBench](https://modelscope.cn/datasets/evalscope/LawBench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2309.16289) |
| **Tags** | `Knowledge`, `QA`, `Reasoning` |
| **Metrics** | `lawbench_score`, `abstention_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `zero_shot` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 9,500 |
| Prompt Length (Mean) | 403.92 chars |
| Prompt Length (Min/Max) | 32 / 23490 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `1-1` | 500 | 40.74 | 32 | 55 |
| `1-2` | 500 | 219.41 | 106 | 503 |
| `2-2` | 500 | 823.05 | 316 | 4524 |
| `2-3` | 500 | 276.44 | 225 | 498 |
| `2-4` | 500 | 213.21 | 181 | 357 |
| `2-5` | 500 | 756.01 | 455 | 1441 |
| `2-6` | 500 | 151.76 | 95 | 375 |
| `2-7` | 500 | 350.63 | 69 | 396 |
| `2-8` | 500 | 484.33 | 282 | 1045 |
| `2-9` | 500 | 215.31 | 167 | 474 |
| `2-10` | 500 | 157.27 | 98 | 377 |
| `3-1` | 500 | 679.97 | 127 | 22218 |
| `3-2` | 500 | 113.21 | 73 | 196 |
| `3-3` | 500 | 585.15 | 103 | 9861 |
| `3-4` | 500 | 684.82 | 146 | 22241 |
| `3-5` | 500 | 1007.58 | 243 | 23490 |
| `3-6` | 500 | 293.43 | 135 | 939 |
| `3-7` | 500 | 535.58 | 136 | 2053 |
| `3-8` | 500 | 86.62 | 79 | 129 |

## Sample Example

**Subset**: `1-1`

```json
{
  "input": [
    {
      "id": "8a9e83b6",
      "content": "回答以下问题，只需直接给出法条内容:\n民法商法农民专业合作社法第三十三条的内容是什么？"
    }
  ],
  "target": "答案:农民专业合作社设理事长一名，可以设理事会。理事长为本社的法定代表人。,农民专业合作社可以设执行监事或者监事会。理事长、理事、经理和财务会计人员不得兼任监事。,理事长、理事、执行监事或者监事会成员，由成员大会从本社成员中选举产生，依照本法和章程的规定行使职权，对成员大会负责。,理事会会议、监事会会议的表决，实行一人一票。",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task": "1-1"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets lawbench \
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
    datasets=['lawbench'],
    dataset_args={
        'lawbench': {
            # subset_list: ['1-1', '1-2', '2-2']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
