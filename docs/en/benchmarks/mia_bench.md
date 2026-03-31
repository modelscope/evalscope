# MIA-Bench


## Overview

MIA-Bench is a multimodal instruction-following benchmark designed to evaluate vision-language models on their ability to follow complex, compositional instructions grounded in images. Each sample contains an image paired with a multi-component instruction, and model responses are scored by an LLM judge per component.

## Task Description

- **Task Type**: Multimodal Instruction Following
- **Input**: Image + multi-component instruction
- **Output**: Free-form response following all instruction components
- **Domains**: Visual understanding, instruction following, language generation

## Key Features

- 400 test samples with diverse instruction types (basic to advanced)
- Each instruction decomposes into 1–5 graded components with weighted scores
- Component types include: describe, length_limit, linguistics, format, etc.
- LLM-as-judge scoring: judge evaluates each component independently and gives a weighted total score (0–10 range, normalized to 0–1)
- No predefined reference answers; scoring is fully judge-based

## Evaluation Notes

- Default evaluation uses the **test** split (400 samples)
- Primary metric: **total_score** (mean of per-sample normalized 0–1 total scores)
- Requires a capable LLM judge (e.g., GPT-4o, Qwen-Max) configured via `judge_model_args`
- Judge strategy should be set to `JudgeStrategy.LLM`


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mia_bench` |
| **Dataset ID** | [lmms-lab/MIA-Bench](https://modelscope.cn/datasets/lmms-lab/MIA-Bench/summary) |
| **Paper** | N/A |
| **Tags** | `InstructionFollowing`, `MultiModal`, `QA` |
| **Metrics** | `total_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 400 |
| Prompt Length (Mean) | 137.81 chars |
| Prompt Length (Min/Max) | 34 / 327 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 400 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 135x240 - 3264x4928 |
| Formats | jpeg, mpo, webp |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "9156e81c",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~195.9KB]"
        },
        {
          "text": "Explain the activity taking place in the image using exactly two sentences, including one metaphor."
        }
      ]
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "instruction": "Explain the activity taking place in the image using exactly two sentences, including one metaphor.",
    "type": "advanced",
    "num_of_component": 3,
    "components": [
      "Explain the activity taking place in the image",
      "using exactly two sentences",
      "including one metaphor"
    ],
    "component_weight": [
      4,
      3,
      3
    ],
    "component_type": [
      "describe",
      "length_limit",
      "linguistics"
    ]
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
    --datasets mia_bench \
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
    datasets=['mia_bench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


