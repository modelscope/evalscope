# VisualOverload


## Overview

VisualOverload (CVPR 2026) probes visual understanding in dense, high-resolution scenes.
Ground-truth answers are privately held; official grading uses the benchmark's evaluation server.

## Task Description

- **Task Type**: Visual question answering
- **Input**: A high-resolution painting and the official `default_prompt`
- **Output**: An option letter, a count, or an OCR answer, depending on the question
- **Domain**: Activities, attributes, counting, OCR, reasoning, and scene understanding

## Key Features

- 2,720 questions from `evalscope/visualoverload` on ModelScope, mirrored from the official
  `paulgavrikov/visualoverload` on Hugging Face
- Combines multiple-choice and free-form questions across easy, medium, and hard difficulties
- Uses the official prompt unchanged and preserves original image bytes by default
- Exports raw responses in the official `question_id`/`response` submission format

## Evaluation Notes

- The dataset is loaded from the ModelScope mirror's public `test` split with zero-shot evaluation.
- `response_present` is a diagnostic for non-empty responses, not answer accuracy.
- Official scores are available only from the evaluation server; no local quality metric is selected.
- Upload `visualoverload_submission.json` from the model's reports directory to
  [the official evaluator](https://huggingface.co/spaces/paulgavrikov/visualoverload-submit).
- Hugging Face sign-in is required for official evaluation; EvalScope does not submit automatically.
- Use one generation per question (`repeats=1`). Duplicate question IDs are rejected during export.
- Limited runs produce partial submissions for testing, not full benchmark scores.
- Model processors may resize or tile images; report those settings. `max_image_bytes` can explicitly
  enable compression, which changes the visual input.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `visualoverload` |
| **Dataset ID** | [evalscope/visualoverload](https://modelscope.cn/datasets/evalscope/visualoverload/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2509.25339) |
| **Tags** | `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `response_present` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,720 |
| Prompt Length (Mean) | 116.59 chars |
| Prompt Length (Min/Max) | 40 / 356 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,720 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 922x2500 - 3412x2432 |
| Formats | jpeg |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "a4c42212",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~3.3MB]"
        },
        {
          "text": "Does the scene feature fantasy characters? Options:\nA. yes\nB. no\nAnswer with the option's letter from the given choices directly."
        }
      ]
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "4ba66530",
    "question_type": "choice",
    "category": "scene",
    "difficulty": "medium"
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
    --datasets visualoverload \
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
    datasets=['visualoverload'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
