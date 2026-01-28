# Winogrande


## Overview

Winogrande is a large-scale benchmark for commonsense reasoning, specifically designed to test pronoun resolution in the Winograd Schema Challenge format. It contains 44K problems that require understanding of physical and social commonsense.

## Task Description

- **Task Type**: Pronoun Resolution / Commonsense Reasoning
- **Input**: Sentence with ambiguous pronoun and two options
- **Output**: Correct option (A or B) that resolves the pronoun
- **Format**: Binary choice between two noun phrases

## Key Features

- 44K Winograd-style pronoun resolution problems
- Adversarially filtered to reduce dataset biases
- Tests physical commonsense (object properties, actions)
- Tests social commonsense (intentions, emotions)
- Requires understanding context to resolve ambiguity

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Binary choice format (option1 vs option2)
- Answers are converted to A/B letter format
- Simple accuracy metric for evaluation
- Commonly used for commonsense reasoning assessment


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `winogrande` |
| **Dataset ID** | [AI-ModelScope/winogrande_val](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,267 |
| Prompt Length (Mean) | 306.58 chars |
| Prompt Length (Min/Max) | 271 / 384 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "f64dd7e2",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B.\n\nSarah was a much better surgeon than Maria so _ always got the easier cases.\n\nA) Sarah\nB) Maria"
    }
  ],
  "choices": [
    "Sarah",
    "Maria"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "winogrande_xs val 0"
  }
}
```

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
    --datasets winogrande \
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
    datasets=['winogrande'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


