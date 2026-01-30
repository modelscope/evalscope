# SciQ

## 概述

SciQ 是一个众包的科学考试题目数据集，涵盖物理、化学、生物及其他科学领域。大多数题目包含支持性的证据段落。

## 任务描述

- **任务类型**：科学问答（多项选择题）
- **输入**：一道包含 4 个选项的科学问题
- **输出**：正确答案的字母（A、B、C 或 D）
- **领域**：物理、化学、生物、地球科学等

## 主要特点

- 众包收集的科学考试题目  
- 覆盖多个科学领域  
- 提供支持性证据段落  
- 考察科学知识与推理能力  
- 适用于科学理解能力评估  

## 评估说明

- 默认配置使用 **0-shot** 评估  
- 使用简单的多项选择提示方式  
- 在测试集（test split）上进行评估  
- 使用简单准确率（accuracy）作为评估指标  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `sciq` |
| **数据集ID** | [extraordinarylab/sciq](https://modelscope.cn/datasets/extraordinarylab/sciq/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `ReadingComprehension` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,000 |
| 提示词长度（平均） | 322.68 字符 |
| 提示词长度（最小/最大） | 244 / 505 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "a30097af",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nCompounds that are capable of accepting electrons, such as o 2 or f2, are called what?\n\nA) antioxidants\nB) Oxygen\nC) residues\nD) oxidants"
    }
  ],
  "choices": [
    "antioxidants",
    "Oxygen",
    "residues",
    "oxidants"
  ],
  "target": "D",
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
    --datasets sciq \
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
    datasets=['sciq'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```