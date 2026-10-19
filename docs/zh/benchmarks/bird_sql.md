# BIRD-SQL Mini-Dev


## 概述

BIRD（面向大规模数据库的文本到SQL基准测试）评估模型是否能够将自然语言问题转换为针对真实、大型且“脏”数据库的SQL查询。此基准测试使用了从11个数据库中选择的500个高质量SQLite实例，即BIRD Mini-Dev。

## 任务描述

- **任务类型**: 文本到SQL
- **输入**: 数据库的`CREATE TABLE`语句、问题以及BIRD提供的外部知识（证据）
- **输出**: 用Markdown代码块包裹的单个SQLite查询
- **领域**: 11个真实的数据库，涵盖金融、体育、医疗保健、教育等领域

## 主要特点

- 500个问题分为三个难度级别：简单148个，中等250个，困难102个
- 真实数据库包含噪声值；外部知识解释业务定义
- 通过执行预测和参考查询进行评分，而不是通过字符串匹配

## 评估说明

- `accuracy`（主要指标）是BIRD执行准确率（EX）：当预测返回的行集与参考SQL返回的行集相同时，预测被认为是正确的，如官方`evaluation_ex.py`所示（忽略行顺序和重复项；列顺序重要）
- 查询以只读方式运行，每个查询一个语句，并有30秒的超时时间（`timeout`）；写入数据库、附加数据库或超过超时时间的查询得分为0
- 提示遵循官方基线（模式 + 问题 + 外部知识 + “逐步思考”），但要求最终查询在代码块中，以便推理模型可以先思考；`sample_rows`添加每个表的示例行，类似于官方的`num_rows`选项
- 数据库（`minidev.zip`，约800 MB）在首次使用时下载并缓存；设置`db_root`为现有的`dev_databases`目录以跳过下载
- 子集是难度级别；`limit`适用于每个子集
- 样本元数据记录`pred_error` / `gold_error`（查询失败的原因）和`gold_empty`（参考返回无行，因此任何返回无行的查询也匹配）
- [论文](https://arxiv.org/abs/2305.03111) | [代码](https://github.com/bird-bench/mini_dev) | [数据集](https://huggingface.co/datasets/birdsql/bird_mini_dev) | [网站](https://bird-bench.github.io/)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bird_sql` |
| **数据集ID** | [evalscope/bird_mini_dev](https://modelscope.cn/datasets/evalscope/bird_mini_dev/summary) |
| **论文** | [论文](https://arxiv.org/abs/2305.03111) |
| **标签** | `Coding`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `mini_dev_sqlite` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 500 |
| 提示词长度（平均） | 4223.89 字符 |
| 提示词长度（最小/最大） | 1343 / 8012 字符 |

**按子集统计:**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `simple` | 148 | 4293.42 | 1343 | 7793 |
| `moderate` | 250 | 4256.66 | 1380 | 7891 |
| `challenging` | 102 | 4042.72 | 1354 | 8012 |

## 样例示例

**子集**: `simple`

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

*注：部分内容为了显示而被截断。*

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
| `sample_rows` | `int` | `0` | 添加到模式提示中的每个表的示例行数（官方默认值：0）。 |

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
