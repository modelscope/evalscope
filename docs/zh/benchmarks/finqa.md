# FinQA


## 概述

FinQA 评估模型在财务报告上的数值推理能力：每个问题针对一家标普 500 公司财报中的一页（文本和一张表格），
答案由金融专家编写的、由算术运算和表格运算组成的简短程序计算得到。

## 任务描述

- **任务类型**：基于文本和表格的数值推理
- **输入**：表格前后的报告文本、表格和一个问题
- **输出**：推理过程，然后是一行用 FinQA DSL 写的 `Program:`，例如 `subtract(5829, 5735), divide(#0, 5735)`
- **领域**：财务报告（标普 500 公司的财报）

## 主要特点

- 来自官方公开测试集的 1,147 个问题（包含标准程序和答案）
- 程序使用 `add`、`subtract`、`multiply`、`divide`、`exp`、`greater` 以及表格运算
  `table_max`、`table_min`、`table_sum`、`table_average`；多数问题需要一到三步

## 评估说明

- 指标遵循官方 `evaluate.py`：`execution_accuracy`（主指标）在问题对应的表格上执行预测程序，并将结果与
  标准答案精确比较（两者都四舍五入到 5 位小数）；`program_accuracy` 检查程序是否与标准程序符号等价
- 评分由官方代码移植实现，每个问题的结果与官方一致
- 执行结果需精确匹配，因此百分比要按比值计算（`divide(#0, 5735)`），与标准程序一致
- 步骤和参数之间用 `, ` 分隔，与数据集一致；括号和分隔符周围多余的空白会被规范化，`op(a,b)` 按 `op(a, b)`
  读取。与官方 DSL 一样，不支持嵌套运算
- 提供整页内容（表格前后的文本和表格），不做检索
- [论文](https://aclanthology.org/2021.emnlp-main.300/) | [代码和数据](https://github.com/czyssrs/FinQA)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `finqa` |
| **数据集ID** | [FinQA](https://github.com/czyssrs/FinQA) |
| **论文** | [论文](https://aclanthology.org/2021.emnlp-main.300/) |
| **标签** | `Math`, `QA`, `Reasoning` |
| **指标** | `execution_accuracy`, `program_accuracy` |
| **默认示例** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,147 |
| 提示词长度（平均） | 5028.46 字符 |
| 提示词长度（最小/最大） | 1356 / 10116 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "5827419a",
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

*注：部分内容为显示而截断。*

## 提示模板

**提示模板：**
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

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets finqa \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['finqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
