# DROP

## 概述

DROP（Discrete Reasoning Over Paragraphs，段落离散推理）是一个具有挑战性的阅读理解基准测试，要求模型在文本段落上执行离散推理操作。与简单的抽取式问答不同，DROP 的问题需要进行数值推理、计数和比较等操作。

## 任务描述

- **任务类型**：带离散推理的阅读理解  
- **输入**：需要推理的段落和问题  
- **输出**：数值答案、文本片段（span）或日期  
- **推理类型**：加法、减法、计数、比较、排序  

## 主要特点

- 包含 96,567 个需要对文本进行离散推理的问题  
- 问题基于 NFL 比赛摘要、维基百科文章等  
- 需要多步推理和算术运算  
- 支持多种有效答案格式（数字、文本片段、日期）  
- 测试模型的组合推理能力  

## 评估说明

- 默认配置使用 **3-shot** 示例  
- 评估指标包括精确匹配（Exact Match, EM）和 token 级别的 F1 分数  
- 答案应遵循格式："Answer: [ANSWER]"  
- F1 分数是主要的比较指标  
- 答案会与多个参考答案进行比对验证  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `drop` |
| **数据集ID** | [AI-ModelScope/DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/summary) |
| **论文** | N/A |
| **标签** | `Reasoning` |
| **指标** | `em`, `f1` |
| **默认示例数量** | 3-shot |
| **评估划分** | `validation` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 9,536 |
| 提示词长度（平均） | 5454.05 字符 |
| 提示词长度（最小/最大） | 4638 / 9893 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "d4ab7ff6",
      "content": "You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.\n\n# Examples\n---\nPassage: Trunajaya rebellion  or Trunajaya War was the ultimately unsuccessful rebellion waged by the Madurese pr ... [TRUNCATED] ... iled a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt.\nQuestion: Who scored the first touchdown of the game?\n\nThink step by step, then write a line of the form \"Answer: [ANSWER]\" at the end of your response."
    }
  ],
  "target": "[('Chaz Schilens',), ('JaMarcus Russell',)]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "passage": " Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.  Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pa ... [TRUNCATED] ...  29-yard touchdown pass from Russell, followed up by an 80-yard punt return for a touchdown.  The Texans tried to rally in the fourth quarter as Brown nailed a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt.",
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

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
You will be asked to read a passage and answer a question. {drop_examples}
# Your Task

---
{query}

Think step by step, then write a line of the form "Answer: [ANSWER]" at the end of your response.
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