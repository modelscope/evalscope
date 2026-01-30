# DROP


## Overview

DROP (Discrete Reasoning Over Paragraphs) is a challenging reading comprehension benchmark that requires models to perform discrete reasoning operations over text passages. Unlike simple extractive QA, DROP questions require numerical reasoning, counting, and comparison operations.

## Task Description

- **Task Type**: Reading Comprehension with Discrete Reasoning
- **Input**: Passage and question requiring reasoning
- **Output**: Numerical answer, span, or date
- **Reasoning Types**: Addition, subtraction, counting, comparison, sorting

## Key Features

- 96,567 questions requiring discrete reasoning over text
- Questions based on NFL game summaries, Wikipedia articles, etc.
- Requires multi-step reasoning and arithmetic operations
- Multiple valid answer formats (numbers, spans, dates)
- Tests compositional reasoning abilities

## Evaluation Notes

- Default configuration uses **3-shot** examples
- Metrics include Exact Match (EM) and token-level F1 score
- Answers should follow the format: "Answer: [ANSWER]"
- F1 score is the primary metric for comparison
- Validates answers against multiple reference answers


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `drop` |
| **Dataset ID** | [AI-ModelScope/DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/summary) |
| **Paper** | N/A |
| **Tags** | `Reasoning` |
| **Metrics** | `em`, `f1` |
| **Default Shots** | 3-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 9,536 |
| Prompt Length (Mean) | 5454.05 chars |
| Prompt Length (Min/Max) | 4638 / 9893 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "d4ab7ff6",
      "content": "You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.\n\n# Examples\n---\nPassage: Trunajaya rebellion  or Trunajaya War was the ultimately unsuccessful rebellion waged by the Madurese pr ... [TRUNCATED] ... iled a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt.\nQuestion: Who scored the first touchdown of the game?\n\nThink step by step, then write a line of the form \"Answer: [ANSWER]\" at the end of your response."
    }
  ],
  "target": "[('Chaz Schilens',), ('JaMarcus Russell',)]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "passage": " Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.  Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pa ... [TRUNCATED] ...  29-yard touchdown pass from Russell, followed up by an 80-yard punt return for a touchdown.  The Texans tried to rally in the fourth quarter as Brown nailed a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt.",
    "answer": {
      "number": "",
      "date": {
        "day": "",
        "month": "",
        "year": ""
      },
      "spans": [
        "Chaz Schilens"
      ],
      "worker_id": "",
      "hit_id": ""
    },
    "validated_answers": {
      "number": [
        "",
        ""
      ],
      "date": [
        {
          "day": "",
          "month": "",
          "year": ""
        },
        {
          "day": "",
          "month": "",
          "year": ""
        }
      ],
      "spans": [
        [
          "Chaz Schilens"
        ],
        [
          "JaMarcus Russell"
        ]
      ],
      "worker_id": [
        "",
        ""
      ],
      "hit_id": [
        "",
        ""
      ]
    }
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
You will be asked to read a passage and answer a question. {drop_examples}
# Your Task

---
{query}

Think step by step, then write a line of the form "Answer: [ANSWER]" at the end of your response.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets drop \
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
    datasets=['drop'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


