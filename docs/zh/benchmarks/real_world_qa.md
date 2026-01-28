# RealWorldQA


## 概述

RealWorldQA 是由 XAI 贡献的一项基准测试，旨在评估多模态 AI 模型对现实世界空间和物理环境的理解能力。该基准使用日常场景中的真实图像来测试模型的实际视觉理解能力。

## 任务描述

- **任务类型**：现实世界视觉问答（Real-World Visual Question Answering）
- **输入**：包含空间/物理问题的真实世界图像
- **输出**：关于场景的可验证答案
- **领域**：物理环境、驾驶场景、日常生活场景

## 主要特点

- 包含 700 多张来自现实场景的图像
- 包含车载摄像头拍摄的图像（驾驶场景）
- 问题具有可验证的标准答案
- 测试空间理解与物理推理能力
- 评估 AI 在实际应用中的理解能力

## 评估说明

- 默认配置采用 **0-shot** 评估方式
- 答案需遵循 "ANSWER: [ANSWER]" 格式
- 使用逐步推理提示（step-by-step reasoning prompting）
- 采用简单准确率（accuracy）作为评估指标
- 在实际、现实世界的场景中测试模型能力


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `real_world_qa` |
| **数据集 ID** | [lmms-lab/RealWorldQA](https://modelscope.cn/datasets/lmms-lab/RealWorldQA/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `acc` |
| **默认样本数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 765 |
| 提示词长度（平均） | 554.79 字符 |
| 提示词长度（最小/最大） | 459 / 904 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 765 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 626x418 - 1536x1405 |
| 格式 | webp |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "6492d8ea",
      "content": [
        {
          "text": "Read the picture and solve the following problem step by step.The last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the problem.\n\nIn which direction is the front wheel of the  ... [TRUNCATED] ... e letter of the correct option and nothing else.\n\nRemember to put your answer on its own line at the end in the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the problem, and you do not need to use a \\boxed command."
        },
        {
          "image": "[BASE64_IMAGE: webp, ~810.4KB]"
        }
      ]
    }
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "image_path": "0.webp"
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
Read the picture and solve the following problem step by step.The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem, and you do not need to use a \boxed command.
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets real_world_qa \
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
    datasets=['real_world_qa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```