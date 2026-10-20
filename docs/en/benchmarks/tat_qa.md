# TAT-QA


## Overview

TAT-QA (Tabular And Textual dataset for Question Answering) evaluates numerical reasoning over hybrid
financial data: each question is asked about a table from a real financial report together with at least
two paragraphs describing it. Answers are text spans, sets of spans, counts or numbers that must be
computed with arithmetic, and most carry a scale (thousand, million, billion or percent).

## Task Description

- **Task Type**: Table + text question answering with numerical reasoning
- **Input**: A table, its paragraphs and a question
- **Output**: Reasoning, then an `Answer:` line and a `Scale:` line
- **Domain**: Financial reports

## Key Features

- 1,663 test questions over 277 tables, with the gold answers released by the authors
- Subsets by where the answer comes from: `table` (736), `text` (381) and `table-text` (546)
- Answer types: span (714), arithmetic (699), multi-span (210) and count (40)
- Scales: none (836), thousand (319), percent (294) and million (214)

## Evaluation Notes

- Metrics are the official exact match (`exact_match`) and numeracy-focused F1 (`f1`, primary), computed with a port of
  `tatqa_metric.py` that gives the same per-question scores as the official script
- Numbers are compared after applying the scale and rounding, so `1.2` with scale `million` and `1,200,000`
  match; the official F1 equals EM for arithmetic and count questions
- Multiple spans are separated by `|` on the `Answer:` line; an unknown or missing scale counts as no scale
- Questions are answered zero-shot from the full table and paragraphs (no retrieval)
- Data is mirrored on ModelScope (`evalscope/TAT-QA`); the original release is on Hugging Face (`next-tat/TAT-QA`)
- [Paper](https://aclanthology.org/2021.acl-long.254/) | [Code](https://github.com/NExTplusplus/TAT-QA) |
  [Dataset](https://huggingface.co/datasets/next-tat/TAT-QA) | [Leaderboard](https://nextplusplus.github.io/TAT-QA/)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `tat_qa` |
| **Dataset ID** | [evalscope/TAT-QA](https://modelscope.cn/datasets/evalscope/TAT-QA/summary) |
| **Paper** | [Paper](https://aclanthology.org/2021.acl-long.254/) |
| **Tags** | `Math`, `QA`, `Reasoning` |
| **Metrics** | `exact_match`, `f1` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,663 |
| Prompt Length (Mean) | 2340.89 chars |
| Prompt Length (Min/Max) | 821 / 7405 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `table` | 736 | 2349.84 | 821 | 7073 |
| `text` | 381 | 2527.49 | 833 | 7405 |
| `table-text` | 546 | 2198.61 | 827 | 7387 |

## Sample Example

**Subset**: `table`

```json
{
  "input": [
    {
      "id": "3f044b9c",
      "content": "Below is a table and some paragraphs from a financial report, followed by a question.\n\nTable:\n | June 30, | \n | 2019 | 2018\nDeferred tax assets |  | \nNon-capital loss carryforwards | $161,119 | $129,436\nCapital loss carryforwards | 155 | 417\n ... [TRUNCATED 1480 chars] ... a number, give the number only and put its scale (thousand, million, billion or percent) on the Scale line.\nThink step by step, then end your response with these two lines:\nAnswer: <answer>\nScale: <none, thousand, million, billion or percent>"
    }
  ],
  "target": "948,578",
  "id": 0,
  "group_id": 0,
  "subset_key": "table",
  "metadata": {
    "uid": "107efaa11617ac41f5f9b3b5adf1e98c",
    "answer": [
      "948,578"
    ],
    "answer_type": "span",
    "answer_from": "table",
    "scale": ""
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Below is a table and some paragraphs from a financial report, followed by a question.

Table:
{table}

Paragraphs:
{paragraphs}

Question: {question}

Answer the question using the table and the paragraphs:
- If the answer is text from the table or the paragraphs, copy it exactly. If there are several, separate them with " | ".
- If the answer is a number, give the number only and put its scale (thousand, million, billion or percent) on the Scale line.
Think step by step, then end your response with these two lines:
Answer: <answer>
Scale: <none, thousand, million, billion or percent>
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tat_qa \
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
    datasets=['tat_qa'],
    dataset_args={
        'tat_qa': {
            # subset_list: ['table', 'text', 'table-text']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
