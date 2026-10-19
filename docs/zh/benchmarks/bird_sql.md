# BIRD-SQL Mini-Dev


## 概述

BIRD（面向大规模数据库的大型基准测试）评估模型能否将自然语言问题转换为针对真实、大规模且“脏”的数据库的SQL查询。此基准测试使用了从11个数据库中选出的500个高质量SQLite实例，即BIRD Mini-Dev。

## 任务描述

- **任务类型**: 文本到SQL
- **输入**: 数据库的`CREATE TABLE`语句、问题以及BIRD提供的外部知识（证据）
- **输出**: 一个SQLite查询 ```sql code block
- **Domain**: Databases across 11 domains (finance, sports, healthcare, education, ...)

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
  step"), but asks for the final query in a ```sql block so reasoning models can think first;
  `sample_rows` adds example rows per table as in the official `num_rows` option
- The databases (`minidev.zip`, about 800 MB) are downloaded and cached on first use; set
  `db_root` to an existing `dev_databases` directory to skip the download
- Subsets are the difficulty levels; `limit` applies per subset
- Sample metadata records `pred_error` (why the prediction failed to run) and `gold_empty` (the
  reference returns no rows, so any query returning nothing also matches)
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

*注意：部分内容因显示原因被截断。*

## 提示模板

**提示模板:**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `db_root` | `str | null` | `None` | 包含每个数据库的一个<db_id>/<db_id>.sqlite的目录（minidev.zip中的`dev_databases`文件夹）；如果为null，则下载并缓存minidev.zip。 |
| `timeout` | `float` | `30.0` | 每个查询的执行超时时间（秒）（官方评估器：30）。 |
| `sample_rows` | `int` | `0` | 要添加到模式提示中的每个表的示例行数（官方默认值：0）。 |

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bird_sql \
    --limit 10  # Remove this line for formal evaluation
```

### 使用Python

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
