# VTCBench


## Overview

VTCBench (Vision-Text Compression Benchmark) evaluates long-context understanding when text is represented as
rendered images, and compares it with a pure-text baseline.

## Task Description

- **Task Type**: Long-context question answering with image-based and text-based evaluation modes
- **Input**: Rendered context images plus a question (VTC mode), or the source text plus a question (Text mode)
- **Output**: Short free-form answer
- **Domain**: Retrieval, associative reasoning, and long-term dialogue memory

## Key Features

- Provides matched VTC and Text modes for measuring the effect of vision-text compression
- Includes Retrieval, Reasoning, and Memory subsets derived from RULER, NoLiMa, and LoCoMo
- Uses pre-rendered multi-image documents to preserve the benchmark's visual layouts
- Supports contexts spanning multiple document images

## Evaluation Notes

- Default configuration uses **0-shot** evaluation in VTC mode
- Use `--dataset-args '{"vtcbench": {"extra_params":{"eval_mode":"text"}}}'` to enable the Text baseline
- Retrieval and Reasoning use the official fractional `contains_all` score
- Memory uses the official maximum ROUGE-L F1 across reference answers
- The unified `score` metric dispatches to the official metric for each subset; its report `macro_score` is the
  unweighted mean across the three tasks
- Text mode strips HTML tags and normalizes whitespace in the same way as the official static evaluator
- Content inside `<think>...</think>` is excluded before scoring, matching the official evaluator
- Long-context requests may require a larger model timeout
- If dataset casting reports an offset overflow, set `DATASET_TF_BATCH_SIZE=1`
- [Paper](https://arxiv.org/abs/2512.15649) | [Code](https://github.com/Moenupa/VTCBench)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `vtcbench` |
| **Dataset ID** | [MLLM-CL/VTCBench](https://modelscope.cn/datasets/MLLM-CL/VTCBench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2512.15649) |
| **Tags** | `LongContext`, `MultiModal`, `QA`, `Reasoning`, `Retrieval` |
| **Metrics** | `score`, `contains_all`, `rouge_l` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,200 |
| Prompt Length (Mean) | 236.71 chars |
| Prompt Length (Min/Max) | 89 / 384 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Retrieval` | 800 | 110.38 | 89 | 141 |
| `Reasoning` | 800 | 368.69 | 363 | 384 |
| `Memory` | 600 | 229.15 | 186 | 283 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 26,554 |
| Images per Sample | min: 1, max: 62, mean: 12.07 |
| Resolution Range | 896x896 - 896x896 |
| Formats | jpeg |


## Sample Example

**Subset**: `Retrieval`

```json
{
  "input": [
    {
      "id": "c51f44e8",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~367.1KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~366.1KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~385.2KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~377.7KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~333.5KB]"
        },
        {
          "text": "\n\nQuestion:What are all the special magic numbers for foolish-rawhide mentioned in the provided text?"
        }
      ]
    }
  ],
  "target": "4075987, 5943250",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "problem": "What are all the special magic numbers for foolish-rawhide mentioned in the provided text?",
    "answers": [
      "4075987",
      "5943250"
    ],
    "subset": "Retrieval",
    "eval_mode": "vtc"
  }
}
```

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_mode` | `str` | `vtc` | Evaluation mode: vtc (images+problem) or text (text+problem). Choices: ['vtc', 'text'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets vtcbench \
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
    datasets=['vtcbench'],
    dataset_args={
        'vtcbench': {
            # subset_list: ['Retrieval', 'Reasoning', 'Memory']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
