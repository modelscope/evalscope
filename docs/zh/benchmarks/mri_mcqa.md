# MRI-MCQA

## 概述

MRI-MCQA 是一个专门针对磁共振成像（MRI）的多项选择题基准测试，用于评估 AI 模型对 MRI 物理原理、扫描协议、图像采集及临床应用的理解能力。

## 任务描述

- **任务类型**：医学影像知识多项选择问答
- **输入**：与 MRI 相关的问题及其多个选项
- **输出**：正确答案的字母
- **领域**：医学影像、MRI 物理、放射学

## 主要特点

- 专注于 MRI 技术及其应用
- 考察对 MRI 物理原理和扫描协议的理解
- 涵盖临床 MRI 应用和序列
- 专为评估医学影像 AI 系统而设计
- 采用多项选择格式以实现标准化评估

## 评估说明

- 默认配置使用 **0-shot** 评估
- 在测试集（test split）上进行评估
- 使用简单准确率（accuracy）作为评估指标
- 无训练集可用

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mri_mcqa` |
| **数据集 ID** | [extraordinarylab/mri-mcqa](https://modelscope.cn/datasets/extraordinarylab/mri-mcqa/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `Medical` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 563 |
| 提示词长度（平均） | 457.23 字符 |
| 提示词长度（最小/最大） | 259 / 888 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "84f179d7",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nWhich cardiac chambers are typically imaged on the short-axis view?\n\nA) RA and RV\nB) RA and LA\nC) LA and LV\nD) RV and LV"
    }
  ],
  "choices": [
    "RA and RV",
    "RA and LA",
    "LA and LV",
    "RV and LV"
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
    --datasets mri_mcqa \
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
    datasets=['mri_mcqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```