# DROP


## 概述

DROP（Discrete Reasoning Over Paragraphs，段落离散推理）是一个具有挑战性的阅读理解基准测试，要求模型在文本段落上执行离散推理操作。与简单的抽取式问答不同，DROP 问题需要进行数值推理、计数和比较操作。

## 任务描述

- **任务类型**：包含离散推理的阅读理解
- **输入**：需要推理的段落和问题
- **输出**：一个数字、一个日期、单个文本片段或多个文本片段
- **领域**：NFL 比赛摘要、维基百科文章及其他段落

## 主要特点

- 96,567 个需要对文本进行离散推理的问题
- 需要多步推理，包括加法、减法、计数、比较和排序
- 多种有效的答案格式（数字、片段、日期）
- 测试组合推理能力

## 评估说明

- 默认配置使用 **3-shot** 示例评估验证集
- 以单行 `Answer:` 结束响应。将数字、日期和单个片段写为纯文本，例如 `Answer: 43`
- 对于多个片段，在该行使用字符串 JSON 数组，例如 `Answer: ["Brady", "Palmer"]`
- Exact Match (EM) 忽略顺序比较归一化后的片段。Token-level F1 对齐预测片段和参考片段以最大化重叠
- 每个指标取可用参考答案中的最高分。F1 是主要指标


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `drop` |
| **数据集ID** | [AI-ModelScope/DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/summary) |
| **论文** | N/A |
| **标签** | `Reasoning` |
| **指标** | `exact_match`, `f1` |
| **默认 Shots** | 3-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 9,536 |
| 提示词长度（平均） | 5543.24 字符 |
| 提示词长度（最小/最大） | 4722 / 10010 字符 |

## 样例示例

**子集**: `default`

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

*注意：部分内容为显示目的已截断。*

## 提示模板

**提示模板：**
```text
You will be asked to read a passage and answer a question. {drop_examples}
# Your Task

---
{query}

Think step by step, then write a line of the form "Answer: [ANSWER]" at the end of your response. For multiple spans, write a JSON array of strings on that line, e.g. Answer: ["first span", "second span"].
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets drop \
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
    datasets=['drop'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
