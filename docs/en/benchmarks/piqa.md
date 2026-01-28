# PIQA

## Overview

PIQA (Physical Interaction QA) is a benchmark for evaluating AI models' understanding of physical commonsense - how objects interact in the physical world and what happens when we manipulate them.

## Task Description

- **Task Type**: Physical Commonsense Reasoning
- **Input**: Goal/question with two possible solutions
- **Output**: More physically plausible solution (A or B)
- **Focus**: Physical world knowledge and intuitive physics

## Key Features

- Tests understanding of physical object properties
- Binary choice between plausible/implausible solutions
- Requires intuitive physics reasoning
- Covers everyday physical scenarios
- Adversarially filtered to reduce biases

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses simple multiple-choice prompting
- Evaluates on validation split
- Simple accuracy metric

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `piqa` |
| **Dataset ID** | [extraordinarylab/piqa](https://modelscope.cn/datasets/extraordinarylab/piqa/summary) |
| **Paper** | N/A |
| **Tags** | `Commonsense`, `MCQ`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,838 |
| Prompt Length (Mean) | 426.52 chars |
| Prompt Length (Min/Max) | 220 / 2335 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "a0600392",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B.\n\nHow do I ready a guinea pig cage for it's new occupants?\n ... [TRUNCATED] ... ps, you will also need to supply it with a water bottle and a food dish.\nB) Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish."
    }
  ],
  "choices": [
    "Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.",
    "Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish."
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets piqa \
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
    datasets=['piqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


