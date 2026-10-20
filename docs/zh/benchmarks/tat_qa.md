# TAT-QA


## 概述

TAT-QA（Tabular And Textual dataset for Question Answering）评估模型在表格与文本混合的金融数据上的数值推理能力：
每个问题都针对真实财报中的一张表格及至少两段描述该表格的文字。答案可能是文本片段、多个片段、计数，
或需要通过算术计算得到的数值，且大多带有单位（千、百万、十亿或百分比）。

## 任务描述

- **任务类型**：结合表格与文本、需要数值推理的问答
- **输入**：一张表格、相关段落和一个问题
- **输出**：推理过程，然后是一行 `Answer:` 和一行 `Scale:`
- **领域**：财务报告

## 主要特点

- 测试集共 1,663 个问题，涉及 277 张表格，标准答案由作者公开发布
- 按答案来源划分子集：`table`（736）、`text`（381）和 `table-text`（546）
- 答案类型：单片段 span（714）、算术 arithmetic（699）、多片段 multi-span（210）和计数 count（40）
- 单位：无（836）、千 thousand（319）、百分比 percent（294）和百万 million（214）

## 评估说明

- 指标为官方的精确匹配（`exact_match`）和面向数值的 F1（`f1`，主指标），由 `tatqa_metric.py` 移植实现，
  每个问题的得分与官方脚本一致
- 数值在应用单位并四舍五入后再比较，因此单位为 `million` 的 `1.2` 与 `1,200,000` 视为相同；
  按官方规则，算术题和计数题的 F1 等于 EM
- 多个片段在 `Answer:` 行中用 `|` 分隔；单位缺失或无法识别时按无单位处理
- 零样本作答，直接使用完整的表格和段落（不做检索）
- 官方数据托管在 Hugging Face（`next-tat/TAT-QA`），运行时请设置 `dataset_hub='huggingface'`
- [论文](https://aclanthology.org/2021.acl-long.254/) | [代码](https://github.com/NExTplusplus/TAT-QA) |
  [数据集](https://huggingface.co/datasets/next-tat/TAT-QA) | [排行榜](https://nextplusplus.github.io/TAT-QA/)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tat_qa` |
| **数据集ID** | [next-tat/TAT-QA](https://modelscope.cn/datasets/next-tat/TAT-QA/summary) |
| **论文** | [论文](https://aclanthology.org/2021.acl-long.254/) |
| **标签** | `Math`, `QA`, `Reasoning` |
| **指标** | `exact_match`, `f1` |
| **默认示例** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,663 |
| 提示词长度（平均） | 2340.89 字符 |
| 提示词长度（最小/最大） | 821 / 7405 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
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
      "id": "7ec79d32",
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

*注：部分内容为显示而截断。*

## 提示模板

**提示模板：**
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

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tat_qa \
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
    datasets=['tat_qa'],
    dataset_args={
        'tat_qa': {
            # subset_list: ['table', 'text', 'table-text']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
