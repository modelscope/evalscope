# LawBench


## Overview

LawBench evaluates Chinese legal knowledge through 20 tasks covering memorization, understanding and application. This native adapter follows the [official zero-shot data and scoring code](https://github.com/open-compass/LawBench) at revision `e30981bb3ff54c41571f222e0b23e92d27375388`.

## Task Description

- **Task Type**: Legal question answering, classification, extraction, generation and proofreading
- **Input**: The official instruction followed by the question, separated by a newline
- **Output**: A free-form Chinese response in the task's requested format
- **Domain**: Chinese legal knowledge and legal reasoning

## Key Features

- 10,000 samples: 500 per task; subsets `1-1`–`1-2`, `2-1`–`2-10`, and `3-1`–`3-8`
- Covers article recitation, legal exams, proofreading, dispute classification, reading comprehension, entity/event extraction, summarization, argument mining, judgment prediction, damages calculation and consultation
- Preserves official substring-based answer parsing, character F1, ROUGE-L and normalized imprisonment log distance
- Task `2-1` uses official ChERRANT character-level corpus F0.5, computed once over the evaluated records

## Evaluation Notes

- Install the ordinary scoring dependency with `pip install 'evalscope[lawbench]'`.
- The 20 official zero-shot task files load automatically from [ModelScope](https://modelscope.cn/datasets/evalscope/LawBench). Each file is an unchanged copy from the pinned official revision.
- For offline data, set `dataset_args.lawbench.local_path` to the official checkout root, its `data/zero_shot` directory, or another directory containing the 20 `<task>.json` files.
- Zero-shot only. Select individual tasks with `subset_list`; standard `limit`, `shuffle`, filters and `repeats` apply.
- `lawbench_score` is the task's official score, usually in [0, 1]; imprisonment log-distance scores can be negative. Overall and category scores are unweighted means of represented task scores.
- Officially skipped dispute/imprisonment references are omitted from score denominators but retained in abstention denominators. NER's upstream `anstention_rate` spelling is normalized to `abstention_rate`; the official CSV driver instead silently defaults this misspelled field to zero.
- Task `2-1` additionally requires the official `evaluation/utils` directory and its ChERRANT dependencies (`ltp`, `pypinyin`, `opencc-python-reimplemented`, `timeout-decorator`, `python-Levenshtein`). Set `extra_params.cherrant_path` to that directory and optionally `cherrant_python` to a separate Python interpreter. Scoring runs in private temporary directories; tool errors and timeouts fail explicitly.
- No LLM judge or OpenCompass runtime is required. This is LawBench, distinct from PLawBench.
- When selecting task `2-1` (included by default), add `cherrant_path` to the generic Usage examples below. The complete configuration is:

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='your-model', eval_type='openai_api',
    api_url='http://localhost:8000/v1', api_key='EMPTY',
    datasets=['lawbench'],
    dataset_args={'lawbench': {
        'extra_params': {'cherrant_path': '/path/to/LawBench/evaluation/utils'},
    }},
    generation_config={'temperature': 0, 'max_tokens': 1024},
))
```


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
| Total Samples | 10,000 |
| Prompt Length (Mean) | 391.46 chars |
| Prompt Length (Min/Max) | 32 / 23490 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `1-1` | 500 | 40.74 | 32 | 55 |
| `1-2` | 500 | 219.41 | 106 | 503 |
| `2-1` | 500 | 154.61 | 95 | 527 |
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
      "id": "ac43c78e",
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

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cherrant_path` | `str` | `` | Official LawBench evaluation/utils directory; required for task 2-1. |
| `cherrant_python` | `str` | `` | Python executable with ChERRANT dependencies; defaults to the current interpreter. |
| `cherrant_timeout` | `float` | `600.0` | Timeout in seconds for each ChERRANT subprocess. |

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
            # subset_list: ['1-1', '1-2', '2-1']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
