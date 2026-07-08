# OfficeQA


## Overview

OfficeQA is a grounded reasoning benchmark by Databricks, built for evaluating model/agent performance on end-to-end grounded reasoning tasks over U.S. Treasury Bulletin documents (1939-2025).

## Task Description

- **Task Type**: Agent-based Document QA (grep/search over corpus)
- **Input**: A question + access to parsed Treasury Bulletin text files via bash tools
- **Output**: A precise answer (numeric values, text, or structured data)
- **Evaluation Mode**: Agent with bash tool (grep, cat, etc.) over the corpus

## Key Features

- Two subsets: `officeqa_pro` (133 questions, hard, default) and `officeqa_full` (246 questions, easy+hard)
- Corpus: ~900 parsed Treasury Bulletin text files (~460MB total)
- Agent uses bash tools (grep, cat, head, etc.) to search the corpus
- Scoring uses fuzzy numeric matching with configurable tolerance (1% default)

## Evaluation Notes

- The agent is given access to parsed .txt files in a corpus directory
- Each question's `source_files` field indicates which document(s) contain the answer
- Uses **rule-based scoring** adapted from official reward.py
- Numerical answers matched with 1% relative error tolerance
- Text answers use case-insensitive substring matching


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `officeqa` |
| **Dataset ID** | [evalscope/officeqa](https://modelscope.cn/datasets/evalscope/officeqa/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `Knowledge`, `QA` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 133 |
| Prompt Length (Mean) | 443.06 chars |
| Prompt Length (Min/Max) | 165 / 1186 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "a6357de6",
      "content": "What were the total expenditures (in millions of nominal dollars) for U.S national defense in the calendar year of 1940?\nPlease provide a precise and concise answer."
    }
  ],
  "target": "2,602",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "uid": "UID0001",
    "source_files": "treasury_bulletin_1941_01.txt",
    "difficulty": "hard"
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
    --datasets officeqa \
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
    datasets=['officeqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
