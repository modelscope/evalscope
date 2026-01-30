# Med-MCQA

## 概述

MedMCQA 是一个大规模的多项选择题问答数据集，旨在解决真实世界中的医学入学考试题目。该数据集包含超过 19.4 万道题目，涵盖印度医学入学考试（AIIMS、NEET-PG）中的各类医学主题。

## 任务描述

- **任务类型**：医学知识多项选择题问答（MCQA）
- **输入**：一道包含 4 个选项的医学问题
- **输出**：正确答案对应的字母
- **领域**：临床医学、基础科学、医疗保健

## 主要特点

- 超过 194,000 道医学考试题目
- 来自 AIIMS 和 NEET-PG 考试的真实题目
- 覆盖 21 个医学科目（解剖学、药理学、病理学等）
- 专家验证的正确答案及解析
- 考察医学知识理解与临床推理能力

## 评估说明

- 默认配置使用 **0-shot** 评估
- 在验证集（validation split）上进行评估
- 使用简单准确率（accuracy）作为评估指标
- 提供训练集（train split），可用于少样本（few-shot）学习

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `med_mcqa` |
| **数据集 ID** | [extraordinarylab/medmcqa](https://modelscope.cn/datasets/extraordinarylab/medmcqa/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认样本数** | 0-shot |
| **评估集** | `validation` |
| **训练集** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,183 |
| 提示词长度（平均） | 374.31 字符 |
| 提示词长度（最小/最大） | 232 / 1004 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "9bf68985",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nWhich of the following is not true for myelinated ner ... [TRUNCATED] ... ugh myelinated fibers is slower than non-myelinated fibers\nB) Membrane currents are generated at nodes of Ranvier\nC) Saltatory conduction of impulses is seen\nD) Local anesthesia is effective only when the nerve is not covered by myelin sheath"
    }
  ],
  "choices": [
    "Impulse through myelinated fibers is slower than non-myelinated fibers",
    "Membrane currents are generated at nodes of Ranvier",
    "Saltatory conduction of impulses is seen",
    "Local anesthesia is effective only when the nerve is not covered by myelin sheath"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets med_mcqa \
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
    datasets=['med_mcqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```