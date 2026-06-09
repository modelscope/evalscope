# VQAv2


## 概述

VQAv2 是一个基于 COCO 图像构建的平衡版视觉问答（Visual Question Answering）基准测试，用于评估多模态模型是否能够根据图像内容回答开放式自然语言问题。

## 任务描述

- **任务类型**：开放式视觉问答
- **输入**：图像 + 自然语言问题
- **输出**：简短答案短语
- **领域**：通用图像理解、物体识别、计数、属性、关系等

## 评估说明

- 默认数据源：ModelScope 上的 `lmms-lab/VQAv2`，使用 `validation` 划分
- 主要指标：**VQAv2 软准确率（soft accuracy）**，基于人工标注的答案计算
- 同时报告对可用答案集合的归一化精确匹配（normalized exact match）
- 适配器支持常见答案格式：字符串列表、答案字典列表，或 `multiple_choice_answer`

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `vqav2` |
| **数据集ID** | [lmms-lab/VQAv2](https://modelscope.cn/datasets/lmms-lab/VQAv2/summary) |
| **论文** | [Paper](https://arxiv.org/abs/1612.00837) |
| **标签** | `MultiModal`, `QA` |
| **指标** | `vqa_score`, `exact_match` |
| **默认示例数量** | 0-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 214,354 |
| 提示词长度（平均） | 185.83 字符 |
| 提示词长度（最小/最大） | 165 / 255 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 214,354 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 120x120 - 640x640 |
| 格式 | jpeg, png |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "7410f0b5",
      "content": [
        {
          "text": "Answer the question according to the image using a short phrase.\nWhere is he looking?\nThe last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes)."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~102.7KB]"
        }
      ]
    }
  ],
  "target": "[\"down\", \"down\", \"at table\", \"skateboard\", \"down\", \"table\", \"down\", \"down\", \"down\", \"down\"]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question": "Where is he looking?",
    "answers": [
      "down",
      "down",
      "at table",
      "skateboard",
      "down",
      "table",
      "down",
      "down",
      "down",
      "down"
    ],
    "multiple_choice_answer": "down",
    "question_id": 262148000,
    "question_type": "none of the above",
    "answer_type": "other"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the question according to the image using a short phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes).
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets vqav2 \
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
    datasets=['vqav2'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```