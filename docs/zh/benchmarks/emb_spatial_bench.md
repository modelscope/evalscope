# EmbSpatial-Bench


## 概述

EmbSpatial-Bench 是一个用于评估大视觉语言模型（LVLMs）具身空间理解能力的基准测试。该基准测试从具身场景中自动构建，涵盖以自我为中心视角下的 6 种空间关系：**close**（近）、**far**（远）、**above**（上）、**under**（下）、**left**（左）和 **right**（右）。

## 任务描述

- **任务类型**：多项选择视觉问答（VQA）
- **输入**：一张以自我为中心的 RGB 图像 + 一个包含 4 个候选答案的空间推理问题
- **输出**：单个字母（A / B / C / D），标识正确的物体或空间关系
- **领域**：具身人工智能、空间推理（MP3D 和 AI2Thor 环境）

## 主要特性

- 包含 3,640 个经人工验证的评估问题，源自两个具身环境（MP3D 和 AI2Thor）
- 涵盖 6 类空间关系：close、far、above、under、left、right
- 每个问题需从 4 个选项中选出空间上最准确的答案
- 旨在揭示当前 LVLM 与合格具身智能之间的差距

## 评估说明

- 默认评估使用 **embspatial_bench.json** 文件（共 3,640 个样本）
- 主要指标：**准确率**（`accuracy`）
- 数据集中答案索引为 0 起始（0 → A，1 → B，2 → C，3 → D）
- 图像以 JPEG base64 字符串形式存储在 JSON 文件中
- 子集按 `relation` 字段组织（6 种空间关系类别）
- [论文](https://aclanthology.org/2024.acl-short.33/) | [GitHub](https://github.com/mengfeidu/EmbSpatial-Bench)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `emb_spatial_bench` |
| **数据集 ID** | [evalscope/EmbSpatial-Bench](https://modelscope.cn/datasets/evalscope/EmbSpatial-Bench/summary) |
| **论文** | [Paper](https://aclanthology.org/2024.acl-short.33/) |
| **标签** | `MCQ`, `MultiModal`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,640 |
| 提示词长度（平均） | 428.34 字符 |
| 提示词长度（最小/最大） | 324 / 549 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `close` | 612 | 352.54 | 333 | 385 |
| `far` | 594 | 351.68 | 324 | 385 |
| `above` | 596 | 464.61 | 419 | 545 |
| `under` | 602 | 463.35 | 409 | 549 |
| `left` | 616 | 467.86 | 418 | 545 |
| `right` | 620 | 468.47 | 411 | 540 |

**图像统计数据：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 3,640 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 300x300 - 1296x968 |
| 格式 | jpeg |


## 样例示例

**子集**: `close`

```json
{
  "input": [
    {
      "id": "e50e4e4c",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~35.2KB]"
        },
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nAmong the listed objects, which one is closest to your current location in the image?\n\nA) table\nB) towel\nC) door\nD) basket"
        }
      ]
    }
  ],
  "choices": [
    "table",
    "towel",
    "door",
    "basket"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "subset_key": "close",
  "metadata": {
    "question_id": "mp3d_0",
    "relation": "close",
    "data_source": "mp3d"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

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
    --datasets emb_spatial_bench \
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
    datasets=['emb_spatial_bench'],
    dataset_args={
        'emb_spatial_bench': {
            # subset_list: ['close', 'far', 'above']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
