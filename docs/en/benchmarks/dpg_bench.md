# DPG-Bench


## Overview

DPG-Bench evaluates how well generated images follow dense prompts about objects, attributes, and relations.

## Task Description

- **Task Type**: Text-to-image generation
- **Input**: Dense prompt describing a multi-object scene
- **Output**: Image scored through atomic yes/no questions decomposed from the prompt
- **Domain**: Dense prompt semantic alignment

## Key Features

- 1,065 official prompts and 14,392 yes/no questions with dependencies.
- The ModelScope mirror includes the exact prompts. For a local CSV, place the original text files in `prompts/`.

## Evaluation Notes

- Configure a vision judge with `--judge`. Missing or invalid verdicts exclude the image. The official judge is mPLUG.
- Overall score averages dependency-adjusted image scores. Category scores pool raw verdicts across images.
- EvalScope retains all CSV rows and uses every image for category scoring. The reference script skips its first CSV
  row and uses the final crop for category scoring. Use `repeats=4` to generate four images per prompt.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `dpg_bench` |
| **Dataset ID** | [evalscope/DPG-Bench](https://modelscope.cn/datasets/evalscope/DPG-Bench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2403.05135) |
| **Tags** | `TextToImage` |
| **Metrics** | `dpg_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,065 |
| Prompt Length (Mean) | 407.9 chars |
| Prompt Length (Min/Max) | 234 / 891 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "27494d92",
      "content": "An empty space where an invisible man would be, with a pair of horn-rimmed glasses seemingly floating in mid-air, and a pearl bead necklace draped in the space below them. In the space where his hands would be, a smartphone is held, as if being operated by the unseen figure. Around this curious scene, the room appears ordinary, with a couch and a coffee table nearby, upon which rests a scattering of magazines and a remote control."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "partiprompts97",
    "prompt": "An empty space where an invisible man would be, with a pair of horn-rimmed glasses seemingly floating in mid-air, and a pearl bead necklace draped in the space below them. In the space where his hands would be, a smartphone is held, as if being operated by the unseen figure. Around this curious scene, the room appears ordinary, with a couch and a coffee table nearby, upon which rests a scattering of magazines and a remote control.",
    "questions": [
      {
        "proposition_id": 1,
        "question": "Is there an empty space?",
        "dependency": [
          0
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 2,
        "question": "Is there an invisible man?",
        "dependency": [
          1
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 3,
        "question": "Are there glasses?",
        "dependency": [
          1
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 4,
        "question": "Is there a necklace?",
        "dependency": [
          1
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 5,
        "question": "Is there a smartphone?",
        "dependency": [
          1
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 6,
        "question": "Is there a room?",
        "dependency": [
          0
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 7,
        "question": "Is there a couch?",
        "dependency": [
          6
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 8,
        "question": "Is there a coffee table?",
        "dependency": [
          6
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 9,
        "question": "Are there magazines?",
        "dependency": [
          6
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 10,
        "question": "Is there a remote control?",
        "dependency": [
          6
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      "... [TRUNCATED 9 more items] ..."
    ],
    "category": "",
    "tags": [],
    "image_path": ""
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --model-task image_generation \
    --eval-type text2image \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets dpg_bench \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='text2image',
    model_task='image_generation',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['dpg_bench'],
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
