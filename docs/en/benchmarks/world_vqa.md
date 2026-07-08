# WorldVQA


## Overview

WorldVQA is a benchmark designed to evaluate the atomic visual world knowledge of Multimodal Large Language Models (MLLMs). It measures models' ability to ground and name visual entities across a stratified taxonomy, spanning from common head-class objects to long-tail rarities.

## Task Description

- **Task Type**: Visual Entity Recognition / Knowledge QA
- **Input**: Image + question asking to identify a visual entity
- **Output**: Free-form text answer (specific entity name)
- **Domain**: Nature, architecture, culture, products, transportation, entertainment, brands, sports

## Key Features

- 3000 VQA pairs across 8 semantic categories
- Bilingual: English (non-zh) and Chinese (zh)
- Three difficulty levels: easy, medium, hard
- Tests atomic visual knowledge decoupled from reasoning
- Requires precise entity identification (e.g., specific breed, not generic "dog")

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on **train** split (the benchmark data split)
- Primary metric: **Accuracy** via LLM-as-judge
- Supports LLM judge for semantic equivalence checking
- Results reported per category and overall


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `world_vqa` |
| **Dataset ID** | [evalscope/WorldVQA](https://modelscope.cn/datasets/evalscope/WorldVQA/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,000 |
| Prompt Length (Mean) | 38.4 chars |
| Prompt Length (Min/Max) | 5 / 182 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Nature & Environment` | 326 | 33.9 | 7 | 83 |
| `Locations & Architecture` | 512 | 42.03 | 9 | 122 |
| `Culture, Arts & Crafts` | 506 | 33.71 | 7 | 112 |
| `Objects & Products` | 437 | 55.97 | 9 | 182 |
| `Vehicles, Craft & Transportation` | 306 | 30.67 | 8 | 114 |
| `Entertainment, Media & Gaming` | 511 | 30.5 | 5 | 83 |
| `Brands, Logos & Graphic Design` | 260 | 39.1 | 6 | 83 |
| `Sports, Gear & Venues` | 142 | 41.99 | 9 | 100 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 3,000 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 33x65 - 3840x3840 |
| Formats | gif, jpeg, png, webp |


## Sample Example

**Subset**: `Nature & Environment`

```json
{
  "input": [
    {
      "id": "0c08c4e2",
      "content": [
        {
          "text": "What breed of dog is in the picture?"
        },
        {
          "image": "[BASE64_IMAGE: png, ~392.9KB]"
        }
      ]
    }
  ],
  "target": "Greek Hound",
  "id": 0,
  "group_id": 0,
  "subset_key": "Nature & Environment",
  "metadata": {
    "index": 0,
    "category": "Nature & Environment",
    "language": "non-zh",
    "difficulty": "medium"
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
    --datasets world_vqa \
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
    datasets=['world_vqa'],
    dataset_args={
        'world_vqa': {
            # subset_list: ['Nature & Environment', 'Locations & Architecture', 'Culture, Arts & Crafts']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
