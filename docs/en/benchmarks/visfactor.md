# VisFactor


## Overview

VisFactor evaluates foundational visual cognition in multimodal large language models using 20 vision-centric subtests adapted from the Factor-Referenced Cognitive Test (FRCT). It isolates abilities that support higher-level visual reasoning instead of measuring performance on a single downstream task.

## Task Description

- **Task Type**: Visual cognition assessment with binary and short free-form questions
- **Input**: One to four images interleaved with a task-specific instruction
- **Output**: A JSON object containing a boolean, word, number, coordinate pair, or letter answer
- **Domain**: Visualization and spatial processing, perceptual closure, visual memory, and reasoning

## Key Features

- Contains 3,046 rows representing 808 test items across 20 FRCT subtests
- Uses rule-based variants and grouped consistency checks to reduce average chance performance to approximately 2.9%
- Preserves the official zero-shot prompts and their image ordering from the VLMEvalKit implementation
- Covers hidden figures, gestalt completion, visual memory, mental rotation, path finding, paper folding, and related abilities

## Evaluation Notes

- Uses the **test** split from the ModelScope mirror of the official `VisFactor.tsv`
- Extracts the last `{"answer": ...}` object and applies the official category-specific normalization rules
- A logical test item may contain multiple rows and receives credit only when every row is correct
- Reports each subtest's item-level accuracy; the primary score is the unweighted macro-average over represented subtests
- Scoring is deterministic and does not require an LLM judge


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `visfactor` |
| **Dataset ID** | [lmms-lab-encoder/visfactor](https://modelscope.cn/datasets/lmms-lab-encoder/visfactor/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2502.16435) |
| **Tags** | `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,046 |
| Prompt Length (Mean) | 463.45 chars |
| Prompt Length (Min/Max) | 188 / 932 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 6,048 |
| Images per Sample | min: 1, max: 4, mean: 1.99 |
| Resolution Range | 100x100 - 668x911 |
| Formats | jpeg |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "1a4f53fe",
      "content": [
        {
          "text": "Look at the two images:\n\nBelow is the first image, one simple shape:"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~2.6KB]"
        },
        {
          "text": "Below is the second image, a larger, complex pattern:"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~8.5KB]"
        },
        {
          "text": "Task: Decide whether the shape in the first image is hidden anywhere inside the second image. The shape will never be rotated, flipped, or resized. The shape will always be right-side-up and exactly the same size as in the first image.\n\nOutput: Respond with only one word: “TRUE” if it is present, “FALSE” if it is not, in JSON format as follows: {\"answer\": YOUR_ANSWER_HERE}."
        }
      ]
    }
  ],
  "target": "T",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "category_id": "CF1",
    "category_name": "Hidden Figures Test",
    "eval_index": 0,
    "additional": ""
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
    --datasets visfactor \
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
    datasets=['visfactor'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
