# FinQA

## 概述

FinQA 评估财务报告中的数值推理：每个问题都涉及标准普尔 500 公司的收益报告（文本和表格），其答案是通过金融专家编写的简短算术和表格操作程序计算得出的。

## 任务描述

- **任务类型**：文本和表格上的数值推理
- **输入**：表格前后的报告文本、表格以及一个问题
- **输出**：推理，然后是一行 `Program:` 的 FinQA DSL 代码，例如 `subtract(5829, 5735), divide(#0, 5735)`
- **领域**：财务报告（标准普尔 500 公司的收益报告）

## 主要特点

- 来自官方公开测试集的 1,147 个测试问题（包括黄金程序和答案）
- 程序使用 `add`、`subtract`、`multiply`、`divide`、`exp`、`greater` 和表格操作 `table_max`、`table_min`、`table_sum`、`table_average`；大多数需要一到三个步骤

## 评估说明

- 指标遵循官方的 `evaluate.py`：`execution_accuracy`（主要指标）运行预测程序，并将结果与四舍五入到小数点后五位的黄金答案进行精确比较；`program_accuracy` 检查程序在符号上是否与黄金程序相等
- 评分是官方代码的移植版本，给出每题相同的结果
- 执行是精确的，因此百分比必须作为比率计算（如 `divide(#0, 5735)`），正如黄金程序中所示
- 步骤和参数用 `, ` 分隔，如数据集中所示；括号和分隔符周围的多余空白会被规范化，`op(a,b)` 会被读作 `op(a, b)`。嵌套操作无效，正如官方 DSL 中所述
- 提供整个页面（表格前后的文本和表格），无需检索
- [论文](https://aclanthology.org/2021.emnlp-main.300/) | [代码和数据](https://github.com/czyssrs/FinQA)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `finqa` |
| **数据集ID** | [evalscope/finqa](https://modelscope.cn/datasets/evalscope/finqa/summary) |
| **论文** | [论文](https://aclanthology.org/2021.emnlp-main.300/) |
| **标签** | `Math`, `QA`, `Reasoning` |
| **指标** | `execution_accuracy`, `program_accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,147 |
| 提示词长度（平均值） | 5028.46 字符 |
| 提示词长度（最小/最大值） | 1356 / 10116 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "4e0812d8",
      "content": "Below are the text and the table of a financial report, followed by a question.\n\nText:\nentergy corporation and subsidiaries management 2019s financial discussion and analysis a result of the entergy louisiana and entergy gulf states louisiana ... [TRUNCATED 4197 chars] ... guments with \", \", e.g. \"subtract(5829, 5735), divide(#0, 5735)\" for (5829 - 5735) / 5735. A percentage is computed as a ratio, without multiplying by 100.\nThink step by step, then end your response with a line of the form:\nProgram: <program>"
    }
  ],
  "target": "subtract(5829, 5735)",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "ETR/2016/page_23.pdf-2",
    "context": "Text:\nentergy corporation and subsidiaries management 2019s financial discussion and analysis a result of the entergy louisiana and entergy gulf states louisiana business combination , results of operations for 2015 also include two items tha ... [TRUNCATED 3144 chars] ... nual net rate increase at entergy mississippi of $ 16 million , effective february 2015 , as a result of the mpsc order in the june 2014 rate case .\nsee note 2 to the financial statements for a discussion of rate and regulatory proceedings. .",
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

*注意：部分内容因显示原因被截断。*

## 提示模板

**提示模板:**
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
    --limit 10  # Remove this line for formal evaluation
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
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
