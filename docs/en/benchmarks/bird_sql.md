# BIRD-SQL Mini-Dev


## Overview

BIRD (BIg Bench for LaRge-scale Database Grounded Text-to-SQL) evaluates whether a model can turn a
natural-language question into an SQL query over real, large and "dirty" databases. This benchmark
uses BIRD Mini-Dev, 500 high-quality SQLite instances selected from the BIRD dev set over 11
databases.

## Task Description

- **Task Type**: Text-to-SQL
- **Input**: The `CREATE TABLE` statements of the database, the question, and BIRD's external
  knowledge (evidence) for the question
- **Output**: A single SQLite query wrapped in a Markdown code fence
- **Domain**: 11 real databases spanning finance, sports, healthcare, education, and more

## Key Features

- 500 questions in three difficulty levels: simple 148, moderate 250, challenging 102
- Real databases with noisy values; the external knowledge explains business definitions
- Scored by executing the predicted and reference queries, not by string match

## Evaluation Notes

- `accuracy` (primary) is BIRD execution accuracy (EX): the prediction is correct when the set of
  rows it returns equals the set returned by the reference SQL, as in the official
  `evaluation_ex.py` (row order and duplicates are ignored; column order matters)
- Queries run read-only with one statement per query and a 30 s timeout (`timeout`); a query that
  writes, attaches a database, or exceeds the timeout scores 0
- The prompt follows the official baseline (schema + question + external knowledge + "think step by
  step"), but asks for the final query in a fenced code block so reasoning models can think first;
  `sample_rows` adds example rows per table as in the official `num_rows` option
- The databases (`minidev.zip`, about 800 MB) are downloaded and cached on first use; set
  `db_root` to an existing `dev_databases` directory to skip the download
- Subsets are the difficulty levels; `limit` applies per subset
- Sample metadata records `pred_error` / `gold_error` (why a query failed to run) and
  `gold_empty` (the reference returns no rows, so any query returning nothing also matches)
- [Paper](https://arxiv.org/abs/2305.03111) | [Code](https://github.com/bird-bench/mini_dev) |
  [Dataset](https://huggingface.co/datasets/birdsql/bird_mini_dev) | [Website](https://bird-bench.github.io/)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bird_sql` |
| **Dataset ID** | [evalscope/bird_mini_dev](https://modelscope.cn/datasets/evalscope/bird_mini_dev/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2305.03111) |
| **Tags** | `Coding`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `mini_dev_sqlite` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 500 |
| Prompt Length (Mean) | 4223.89 chars |
| Prompt Length (Min/Max) | 1343 / 8012 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `simple` | 148 | 4293.42 | 1343 | 7793 |
| `moderate` | 250 | 4256.66 | 1380 | 7891 |
| `challenging` | 102 | 4042.72 | 1354 | 8012 |

## Sample Example

**Subset**: `simple`

```json
{
  "input": [
    {
      "id": "20a83e4b",
      "content": "CREATE TABLE customers\n(\n    CustomerID INTEGER UNIQUE     not null\n        primary key,\n    Segment    TEXT null,\n    Currency   TEXT null\n)\n\nCREATE TABLE gasstations\n(\n    GasStationID INTEGER    UNIQUE   not null\n        primary key,\n    C ... [TRUNCATED 1035 chars] ...  EUR against customers who pay in CZK = count(Currency = 'EUR') / count(Currency = 'CZK').\n\nGenerate the SQLite for the above question after thinking step by step.\nWrite the final query in a single ```sql code block at the end of your answer."
    }
  ],
  "target": "SELECT CAST(SUM(IIF(Currency = 'EUR', 1, 0)) AS FLOAT) / SUM(IIF(Currency = 'CZK', 1, 0)) AS ratio FROM customers",
  "id": 0,
  "group_id": 0,
  "subset_key": "simple",
  "metadata": {
    "question_id": 1471,
    "db_id": "debit_card_specializing",
    "question": "What is the ratio of customers who pay in EUR against customers who pay in CZK?",
    "evidence": "ratio of customers who pay in EUR against customers who pay in CZK = count(Currency = 'EUR') / count(Currency = 'CZK').",
    "difficulty": "simple"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_root` | `str | null` | `None` | Directory with one <db_id>/<db_id>.sqlite per database (the `dev_databases` folder of minidev.zip); if null, minidev.zip is downloaded and cached. |
| `timeout` | `float` | `30.0` | Per-query execution timeout in seconds (official evaluator: 30). |
| `sample_rows` | `int` | `0` | Number of example rows per table to add to the schema prompt (official default: 0). |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bird_sql \
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
    datasets=['bird_sql'],
    dataset_args={
        'bird_sql': {
            # subset_list: ['simple', 'moderate', 'challenging']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
