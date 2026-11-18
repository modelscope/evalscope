# DROP


## Overview

DROP (Discrete Reasoning Over Paragraphs) is a challenging reading comprehension benchmark that requires models to perform discrete reasoning operations over text passages. Unlike simple extractive QA, DROP questions require numerical reasoning, counting, and comparison operations.

## Task Description

- **Task Type**: Reading Comprehension with Discrete Reasoning
- **Input**: Passage and question requiring reasoning
- **Output**: A number, a date, a single text span, or multiple text spans
- **Domain**: NFL game summaries, Wikipedia articles, and other passages

## Key Features

- 96,567 questions requiring discrete reasoning over text
- Requires multi-step reasoning, including addition, subtraction, counting, comparison, and sorting
- Multiple valid answer formats (numbers, spans, dates)
- Tests compositional reasoning abilities

## Evaluation Notes

- The default configuration evaluates the validation split with **3-shot** examples
- End the response with a single `Answer:` line. Write numbers, dates, and single spans as plain text, such as `Answer: 43`
- For multiple spans, use a JSON array of strings on that line, such as `Answer: ["Brady", "Palmer"]`
- Exact Match (EM) compares normalized spans regardless of order. Token-level F1 aligns predicted and reference spans to maximize overlap
- Each metric takes the highest score across the available reference answers. F1 is the primary metric


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `drop` |
| **Dataset ID** | [AI-ModelScope/DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/summary) |
| **Paper** | N/A |
| **Tags** | `Reasoning` |
| **Metrics** | `exact_match`, `f1` |
| **Default Shots** | 3-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 9,536 |
| Prompt Length (Mean) | 5543.24 chars |
| Prompt Length (Min/Max) | 4722 / 10010 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "1f5a2e22",
      "content": "You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.\n\n# Examples\n---\nPassage: Trunajaya rebellion  or Trunajaya War was the ultimately unsuccessful rebellion waged by the Madurese pr ... [TRUNCATED 4923 chars] ... ed the first touchdown of the game?\n\nThink step by step, then write a line of the form \"Answer: [ANSWER]\" at the end of your response. For multiple spans, write a JSON array of strings on that line, e.g. Answer: [\"first span\", \"second span\"]."
    }
  ],
  "target": "[('Chaz Schilens',), ('JaMarcus Russell',)]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "passage": " Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.  Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pa ... [TRUNCATED 447 chars] ...  29-yard touchdown pass from Russell, followed up by an 80-yard punt return for a touchdown.  The Texans tried to rally in the fourth quarter as Brown nailed a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt.",
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

Think step by step, then write a line of the form "Answer: [ANSWER]" at the end of your response. For multiple spans, write a JSON array of strings on that line, e.g. Answer: ["first span", "second span"].
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
