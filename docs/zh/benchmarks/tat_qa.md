# TAT-QA

## 概述

TAT-QA（用于问答的表格和文本数据集）评估混合财务数据上的数值推理：每个问题都是关于真实财务报告中的一个表格以及至少两段描述该表格的段落。答案可以是文本片段、片段集合、计数或需要通过算术计算得出的数字，并且大多数带有单位（千、百万、十亿或百分比）。

## 任务描述

- **任务类型**：带有数值推理的表格+文本问答
- **输入**：一个表格、其段落和一个问题
- **输出**：推理过程，然后是一行`Answer:`和一行`Scale:`
- **领域**：财务报告

## 主要特点

- 1,663个测试问题涉及277个表格，作者已发布正确答案
- 根据答案来源划分子集：`table`（736）、`text`（381）和`table-text`（546）
- 答案类型：片段（714）、算术（699）、多片段（210）和计数（40）
- 单位：无（836）、千（319）、百分比（294）和百万（214）

## 评估说明

- 指标是官方精确匹配 (`exact_match`) 和以数值为重点的F1 (`f1`, 主要)，使用移植的`tatqa_metric.py`计算，与官方脚本给出的每题得分相同
- 数字在应用单位并四舍五入后进行比较，因此带有单位“百万”的`1.2`和`1,200,000`匹配；对于算术和计数问题，官方F1等于EM
- 多个片段在`Answer:`行中用`|`分隔；未知或缺失的单位视为无单位
- 问题从完整的表格和段落中零样本回答（无需检索）
- 数据在ModelScope上镜像 (`evalscope/TAT-QA`)；原始发布在Hugging Face上 (`next-tat/TAT-QA`)
- [论文](https://aclanthology.org/2021.acl-long.254/) | [代码](https://github.com/NExTplusplus/TAT-QA) | [数据集](https://huggingface.co/datasets/next-tat/TAT-QA) | [排行榜](https://nextplusplus.github.io/TAT-QA/)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tat_qa` |
| **数据集ID** | [evalscope/TAT-QA](https://modelscope.cn/datasets/evalscope/TAT-QA/summary) |
| **论文** | [论文](https://aclanthology.org/2021.acl-long.254/) |
| **标签** | `Math`, `QA`, `Reasoning` |
| **指标** | `exact_match`, `f1` |
| **默认样本数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,663 |
| 提示词长度（平均值） | 2340.89 字符 |
| 提示词长度（最小/最大值） | 821 / 7405 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `table` | 736 | 2349.84 | 821 | 7073 |
| `text` | 381 | 2527.49 | 833 | 7405 |
| `table-text` | 546 | 2198.61 | 827 | 7387 |

## 样例示例

**子集**: `table`

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

*注：部分内容因显示原因被截断。*

## 提示模板

**提示模板:**
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

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tat_qa \
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
