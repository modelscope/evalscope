# OmniBench

## 概述

OmniBench 是一个开创性的通用多模态基准测试，旨在严格评估多模态大语言模型（MLLMs）在视觉、听觉和文本输入上同时进行识别、理解和推理的能力。

## 任务描述

- **任务类型**：通用多模态理解（图像 + 音频 + 文本）
- **输入**：图像、音频和带选项的文本问题
- **输出**：正确答案选项
- **模态**：视觉、听觉和文本

## 核心特性

- 三模态评估（图像、音频、文本）
- 测试跨模态推理能力
- 多选题格式
- 可配置模态使用方式
- 支持思维链（Chain-of-thought）提示

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用简单准确率（accuracy）作为指标
- 额外参数：`use_image`（布尔值）、`use_audio`（布尔值）
- 支持为图像/音频提供文本替代内容

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `omni_bench` |
| **数据集ID** | [m-a-p/OmniBench](https://modelscope.cn/datasets/m-a-p/OmniBench/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,142 |
| 提示词长度（平均） | 477.61 字符 |
| 提示词长度（最小/最大） | 310 / 1280 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 1,142 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 361x203 - 5184x3456 |
| 格式 | jpeg, png |

**音频统计信息：**

| 指标 | 值 |
|--------|-------|
| 音频文件总数 | 1,142 |
| 每样本音频数 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | mp3 |

## 样例示例

**子集**：`default`

```json
{
  "input": [
    {
      "id": "a5cb4fd0",
      "content": [
        {
          "text": "Answer the following multiple choice question based on the image and audio content. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nWhat are the men doing?\n\nA) The man in jeans is taking notes from the newspaper.\nB) The man in purple is reading the newspaper.\nC) The man in jeans is playing a crossword puzzle.\nD) The man on the table is doing a crossword puzzle."
        },
        {
          "image": "[BASE64_IMAGE: png, ~3.7MB]"
        },
        {
          "audio": "[BASE64_AUDIO: mp3, ~103.6KB]",
          "format": "mp3"
        }
      ]
    }
  ],
  "choices": [
    "The man in jeans is taking notes from the newspaper.",
    "The man in purple is reading the newspaper.",
    "The man in jeans is playing a crossword puzzle.",
    "The man on the table is doing a crossword puzzle."
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "task_type": "Action and Activity",
    "audio_type": "speech",
    "answer": "The man in jeans is playing a crossword puzzle.",
    "image_content": "The image depicts a scene from a television show set in a cozy, well-furnished apartment. The main focus is on two individuals sitting on a couch in the foreground. The man on the left is wearing a dark blue jacket and jeans, and he is holdin ... [TRUNCATED] ... ted with a mix of modern and vintage elements, including a colorful rug, patterned curtains, and a variety of decorative items on the shelves and walls. The overall atmosphere is warm and inviting, suggesting a comfortable and lived-in space.",
    "audio_content": "Two people talking: A: Four letters, \"circle or hoop.\" B: Ring! Damn it, ring! A: Thanks."
  }
}
```

*注：部分内容因显示需要已被截断。*

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question based on the image and audio content. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `use_image` | `bool` | `True` | 是否提供原始图像。设为 False 时使用文本替代内容。 |
| `use_audio` | `bool` | `True` | 是否提供原始音频。设为 False 时使用文本替代内容。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets omni_bench \
    --limit 10  # 正式评估时请移除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['omni_bench'],
    dataset_args={
        'omni_bench': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```