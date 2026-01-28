# BioMixQA

## 概述

BioMixQA 是一个精心整理的生物医学问答数据集，旨在评估 AI 模型在生物医学知识与推理方面的能力。该数据集已被用于验证基于知识图谱的检索增强生成（KG-RAG）框架在不同大语言模型（LLMs）上的有效性。

## 任务描述

- **任务类型**：生物医学多选题问答
- **输入**：包含多个选项的生物医学问题
- **输出**：正确答案对应的字母
- **领域**：生物医学、医疗健康、生命科学

## 主要特点

- 问题来源于多样化的生物医学资源
- 考察对医学与生物学知识的理解能力
- 验证 RAG 框架在生物医学领域的有效性
- 采用多选题格式以实现标准化评估
- 适用于评估医疗健康领域的 AI 系统

## 评估说明

- 默认配置使用 **0-shot** 评估方式
- 使用简单准确率（accuracy）作为性能指标
- 在测试集（test split）上进行评估
- 未提供 few-shot 示例

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `biomix_qa` |
| **数据集ID** | [extraordinarylab/biomix-qa](https://modelscope.cn/datasets/extraordinarylab/biomix-qa/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `Medical` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 306 |
| 提示词长度（平均） | 344.93 字符 |
| 提示词长度（最小/最大） | 316 / 393 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "5e830918",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E.\n\nOut of the given list, which Gene is associated with head and neck cancer and uveal melanoma.\n\nA) ABO\nB) CACNA2D1\nC) PSCA\nD) TERT\nE) SULT1B1"
    }
  ],
  "choices": [
    "ABO",
    "CACNA2D1",
    "PSCA",
    "TERT",
    "SULT1B1"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

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
    --datasets biomix_qa \
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
    datasets=['biomix_qa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```