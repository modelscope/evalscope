# BLINK


## Overview

BLINK is a benchmark designed to evaluate the core visual perception abilities of Multimodal Large Language Models (MLLMs). It transforms 14 classic computer vision tasks into 3,807 multiple-choice questions with single or multiple images and visual prompts.

## Task Description

- **Task Type**: Visual Perception Multiple-Choice QA
- **Input**: One or more images + multiple-choice question
- **Output**: Single answer letter
- **Domains**: Visual perception, correspondence, reasoning, detection

## Key Features

- Covers 14 diverse visual perception tasks
- Supports single and multi-image inputs (up to 4 images)
- Tests fundamental visual understanding capabilities
- Categories include: Art Style, Counting, Forensic Detection, IQ Test, Jigsaw, Multi-view Reasoning, Object Localization, and more
- Questions derived from classic computer vision benchmarks

## Evaluation Notes

- Default evaluation uses the **val** split
- Primary metric: **Accuracy** on multiple-choice questions
- Uses "ANSWER: [LETTER]" format for responses
- Results can be analyzed across 14 different perception categories


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `blink` |
| **Dataset ID** | [evalscope/BLINK](https://modelscope.cn/datasets/evalscope/BLINK/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `val` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,901 |
| Prompt Length (Mean) | 577.53 chars |
| Prompt Length (Min/Max) | 252 / 1125 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Art_Style` | 117 | 553 | 553 | 553 |
| `Counting` | 120 | 285.21 | 270 | 317 |
| `Forensic_Detection` | 132 | 480 | 480 | 480 |
| `Functional_Correspondence` | 130 | 1118.34 | 1113 | 1125 |
| `IQ_Test` | 150 | 884.6 | 548 | 922 |
| `Jigsaw` | 150 | 543 | 543 | 543 |
| `Multi-view_Reasoning` | 133 | 549 | 549 | 549 |
| `Object_Localization` | 122 | 531.86 | 527 | 548 |
| `Relative_Depth` | 124 | 359 | 359 | 359 |
| `Relative_Reflectance` | 134 | 498 | 498 | 498 |
| `Semantic_Correspondence` | 139 | 952 | 952 | 952 |
| `Spatial_Relation` | 143 | 263.97 | 252 | 282 |
| `Visual_Correspondence` | 172 | 587 | 587 | 587 |
| `Visual_Similarity` | 135 | 414 | 414 | 414 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 3,675 |
| Images per Sample | min: 1, max: 4, mean: 1.93 |
| Resolution Range | 200x83 - 3072x4096 |
| Formats | jpeg |


## Sample Example

**Subset**: `Art_Style`

```json
{
  "input": [
    {
      "id": "a522940e",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format:\n'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B.\n\nSome most common art painting styles include Realism, Impressi ... [TRUNCATED] ...  of art paintings, use the first image as the reference image, and determine which one of the second or the third image shares the same style as the reference image?\nSelect from the following choices.\n(A) the second image\n(B) the third image\n"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~477.8KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~876.1KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~329.2KB]"
        }
      ]
    }
  ],
  "choices": [
    "the second image",
    "the third image"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format:
'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets blink \
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
    datasets=['blink'],
    dataset_args={
        'blink': {
            # subset_list: ['Art_Style', 'Counting', 'Forensic_Detection']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


