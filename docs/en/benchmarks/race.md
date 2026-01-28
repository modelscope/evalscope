# RACE


## Overview

RACE (ReAding Comprehension from Examinations) is a large-scale reading comprehension benchmark collected from Chinese middle school and high school English examinations. It tests comprehensive reading comprehension abilities.

## Task Description

- **Task Type**: Reading Comprehension (Multiple-Choice)
- **Input**: Article passage with question and 4 answer choices
- **Output**: Correct answer letter (A, B, C, or D)
- **Difficulty Levels**: Middle school and High school

## Key Features

- 28,000+ passages with 100,000 questions
- Real examination questions for authentic difficulty
- Two subsets: middle (easier) and high (harder)
- Tests various comprehension skills (inference, vocabulary, main idea, etc.)
- Diverse article topics and question types

## Evaluation Notes

- Default configuration uses **3-shot** examples
- Maximum few-shot number is 3 (context length consideration)
- Uses Chain-of-Thought (CoT) prompting
- Two subsets available: `high` and `middle`
- Evaluates on test split


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `race` |
| **Dataset ID** | [evalscope/race](https://modelscope.cn/datasets/evalscope/race/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 3-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4,934 |
| Prompt Length (Mean) | 7217.34 chars |
| Prompt Length (Min/Max) | 3685 / 11131 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `high` | 3,498 | 8279.47 | 6585 | 11131 |
| `middle` | 1,436 | 4630.07 | 3685 | 6032 |

## Sample Example

**Subset**: `high`

```json
{
  "input": [
    {
      "id": "706382e9",
      "content": "Here are some examples of how to answer similar questions:\n\nArticle:\nLast week I talked with some of my students about what they wanted to do after they graduated, and what kind of job prospects  they thought they had.\nGiven that I teach stud ... [TRUNCATED] ... I owe my life to her,\" said Nancy with tears.\nQuestion:\nWhat did Nancy try to do before she fell over?\n\nA) Measure the depth of the river\nB) Look for a fallen tree trunk\nC) Protect her cows from being drowned\nD) Run away from the flooded farm"
    }
  ],
  "choices": [
    "Measure the depth of the river",
    "Look for a fallen tree trunk",
    "Protect her cows from being drowned",
    "Run away from the flooded farm"
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "example_id": "high19432.txt"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

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
    --datasets race \
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
    datasets=['race'],
    dataset_args={
        'race': {
            # subset_list: ['high', 'middle']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


