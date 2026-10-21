# OmniBench

## 概述

OmniBench 是一个开创性的通用多模态基准测试，旨在严格评估MLLM在同时识别、解释和推理视觉、听觉和文本输入方面的能力。

## 任务描述

- **任务类型**：通用多模态理解（图像 + 音频 + 文本）
- **输入**：图像、音频和带有选项的文本问题
- **输出**：正确的答案选项
- **模态**：视觉、听觉和文本

## 主要特点

- 三模态评估（图像、音频、文本）
- 测试跨模态推理能力
- 多项选择题格式
- 可配置的模态使用
- 链式思维提示

## 评估说明

- 默认配置使用 **0-shot** 评估
- 简单的准确率指标
- 额外参数：use_image (bool), use_audio (bool)
- 支持图像/音频的文本替代
- 需要支持 `input_audio` 内容部分的OpenAI兼容服务器。SGLang在v0.5.19 (sgl-project/sglang#33606) 中添加了此支持；早期版本会以HTTP 400拒绝音频请求。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `omni_bench` |
| **数据集ID** | [m-a-p/OmniBench](https://modelscope.cn/datasets/m-a-p/OmniBench/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiModal` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,142 |
| 提示词长度（平均） | 477.61 字符 |
| 提示词长度（最小/最大） | 310 / 1280 字符 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 1,142 |
| 每个样本的图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 361x203 - 5184x3456 |
| 格式 | jpeg, png |

**音频统计：**

| 指标 | 值 |
|--------|-------|
| 总音频文件数 | 1,142 |
| 每个样本的音频数 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | mp3 |

## 样例示例

**子集**: `default`

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

*注：部分内容为显示而被截断。*

## 提示模板

**提示模板:**
```text
Answer the following multiple choice question based on the image and audio content. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `use_image` | `bool` | `True` | 是否提供原始图像。False 使用文本替代。 |
| `use_audio` | `bool` | `True` | 是否提供原始音频。False 使用文本替代。 |

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets omni_bench \
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
    datasets=['omni_bench'],
    dataset_args={
        'omni_bench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
