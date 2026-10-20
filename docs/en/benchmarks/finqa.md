# FinQA


## Overview

FinQA evaluates numerical reasoning over financial reports: each question is about one page of an S&P 500
company's earnings report (text and a table), and its answer is computed with a short program of
arithmetic and table operations written by financial experts.

## Task Description

- **Task Type**: Numerical reasoning over text and tables
- **Input**: The report text before and after the table, the table and a question
- **Output**: Reasoning, then a `Program:` line in the FinQA DSL, e.g. `subtract(5829, 5735), divide(#0, 5735)`
- **Domain**: Financial reports (earnings reports of S&P 500 companies)

## Key Features

- 1,147 test questions from the official public test set (gold programs and answers included)
- Programs use `add`, `subtract`, `multiply`, `divide`, `exp`, `greater` and the table operations
  `table_max`, `table_min`, `table_sum`, `table_average`; most need one to three steps

## Evaluation Notes

- Metrics follow the official `evaluate.py`: `execution_accuracy` (primary) runs the predicted program on the
  question's table and compares the result exactly with the gold answer, both rounded to 5 decimals;
  `program_accuracy` checks that the program is symbolically equal to the gold program
- The scoring is a port of the official code and gives the same per-question results
- Execution is exact, so a percentage must be computed as a ratio (`divide(#0, 5735)`), as in the gold programs
- Steps and arguments are separated by `, ` as in the dataset; extra whitespace around brackets and separators
  is normalized, and `op(a,b)` is read as `op(a, b)`. Nested operations are invalid, as in the official DSL
- The whole page (text before and after the table and the table) is given, without retrieval
- [Paper](https://aclanthology.org/2021.emnlp-main.300/) | [Code and data](https://github.com/czyssrs/FinQA)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `finqa` |
| **Dataset ID** | [evalscope/finqa](https://modelscope.cn/datasets/evalscope/finqa/summary) |
| **Paper** | [Paper](https://aclanthology.org/2021.emnlp-main.300/) |
| **Tags** | `Math`, `QA`, `Reasoning` |
| **Metrics** | `execution_accuracy`, `program_accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,147 |
| Prompt Length (Mean) | 5028.46 chars |
| Prompt Length (Min/Max) | 1356 / 10116 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "528cbea9",
      "content": "Below are the text and the table of a financial report, followed by a question.\n\nText:\nentergy corporation and subsidiaries management 2019s financial discussion and analysis a result of the entergy louisiana and entergy gulf states louisiana ... [TRUNCATED 4197 chars] ... guments with \", \", e.g. \"subtract(5829, 5735), divide(#0, 5735)\" for (5829 - 5735) / 5735. A percentage is computed as a ratio, without multiplying by 100.\nThink step by step, then end your response with a line of the form:\nProgram: <program>"
    }
  ],
  "target": "subtract(5829, 5735)",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "ETR/2016/page_23.pdf-2",
    "program": "subtract(5829, 5735)",
    "answer": 94.0,
    "table": [
      [
        "",
        "amount ( in millions )"
      ],
      [
        "2014 net revenue",
        "$ 5735"
      ],
      [
        "retail electric price",
        "187"
      ],
      [
        "volume/weather",
        "95"
      ],
      [
        "waterford 3 replacement steam generator provision",
        "-32 ( 32 )"
      ],
      [
        "miso deferral",
        "-35 ( 35 )"
      ],
      [
        "louisiana business combination customer credits",
        "-107 ( 107 )"
      ],
      [
        "other",
        "-14 ( 14 )"
      ],
      [
        "2015 net revenue",
        "$ 5829"
      ]
    ]
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Below are the text and the table of a financial report, followed by a question.

{context}

Question: {question}

Write a program that computes the answer, using these operations:
- add(a, b), subtract(a, b), multiply(a, b), divide(a, b), exp(a, b) (a to the power b), greater(a, b) (yes if a > b, else no)
- table_max(row, none), table_min(row, none), table_sum(row, none), table_average(row, none): over the numbers in a table row, where row is the row name exactly as in the first column
Arguments are numbers from the report (plain digits, no units, commas or % signs), constants such as const_100 or const_1000 (const_m1 is -1), or #n, the result of step n (counting from 0).
Use one operation per step (no nested operations) and separate steps and arguments with ", ", e.g. "subtract(5829, 5735), divide(#0, 5735)" for (5829 - 5735) / 5735. A percentage is computed as a ratio, without multiplying by 100.
Think step by step, then end your response with a line of the form:
Program: <program>
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets finqa \
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
    datasets=['finqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
