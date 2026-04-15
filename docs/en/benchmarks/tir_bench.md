# TIR-Bench


## Overview

TIR-Bench (Thinking-with-Images Reasoning Benchmark) is a comprehensive multimodal benchmark
that evaluates agentic visual reasoning capabilities of vision-language models. It covers
diverse task categories requiring spatial, compositional, and multi-step visual reasoning.

## Task Description

- **Task Type**: Multi-task Visual Reasoning (MCQ, OCR, word search, spot difference, jigsaw, etc.)
- **Input**: One or two images + question (most tasks use multiple-choice format)
- **Output**: Answer letter (MCQ) or numeric/text response depending on task type
- **Domains**: instrument, color, refcoco, rotation_game, math, word_search, visual_search, ocr, symbolic, spot_difference, contrast, jigsaw, maze

## Key Features

- 1,215 test samples across 13 diverse visual reasoning task categories
- Covers single-image and dual-image reasoning scenarios
- Answers span letter choices (A-J), integers, floats, and text
- Task-specific scoring with LLM-as-judge fallback for robust evaluation

## Evaluation Notes

- Default evaluation uses the **test** split (1,215 samples)
- Primary metric: **Accuracy** (acc)
- Images are downloaded as `data.zip` from ModelScope and extracted automatically
- Rule-based scoring: OCR (substring match), jigsaw (grid IoU), spot_difference (set IoU),
  word_search (numeric match), all other tasks (MCQ / numeric judge)
- **Recommended**: set `judge_strategy=JudgeStrategy.LLM_RECALL` and provide `judge_model_args`
  to activate LLM-as-judge as a recall mechanism — the judge is called only when rule-based
  scoring gives 0, providing more accurate evaluation without unnecessary API overhead
- [Paper](https://arxiv.org/abs/2511.01833) | [GitHub](https://github.com/agents-x-project/TIR-Bench)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `tir_bench` |
| **Dataset ID** | [evalscope/TIR-Bench](https://modelscope.cn/datasets/evalscope/TIR-Bench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2511.01833) |
| **Tags** | `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,215 |
| Prompt Length (Mean) | 384.97 chars |
| Prompt Length (Min/Max) | 19 / 4039 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `instrument` | 80 | 110.96 | 57 | 196 |
| `color` | 100 | 130.26 | 98 | 241 |
| `refcoco` | 120 | 144.51 | 132 | 182 |
| `rotation_game` | 75 | 146.44 | 140 | 148 |
| `math` | 120 | 126.69 | 50 | 397 |
| `word_search` | 100 | 126.72 | 24 | 307 |
| `visual_search` | 120 | 111.64 | 19 | 501 |
| `ocr` | 60 | 35.08 | 29 | 116 |
| `symbolic` | 50 | 88.18 | 66 | 243 |
| `spot_difference` | 100 | 1114.79 | 93 | 1379 |
| `contrast` | 50 | 48.1 | 31 | 123 |
| `jigsaw` | 120 | 605 | 605 | 605 |
| `maze` | 120 | 1527.06 | 626 | 4039 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,255 |
| Images per Sample | min: 1, max: 2, mean: 1.03 |
| Resolution Range | 60x23 - 6944x9280 |
| Formats | jpeg, mpo, png, webp |


## Sample Example

**Subset**: `instrument`

```json
{
  "input": [
    {
      "id": "dd25f10d",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpg, ~2.9MB]"
        },
        {
          "text": "According to the image, what is the thermometer reading in Fahrenheit? Answer as an integer like 1,2,3."
        }
      ]
    }
  ],
  "target": "72",
  "id": 0,
  "group_id": 0,
  "subset_key": "instrument",
  "metadata": {
    "task": "instrument",
    "meta_data": {},
    "id": 6
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
    --datasets tir_bench \
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
    datasets=['tir_bench'],
    dataset_args={
        'tir_bench': {
            # subset_list: ['instrument', 'color', 'refcoco']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


