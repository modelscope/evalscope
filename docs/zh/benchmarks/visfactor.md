# VisFactor


## 概述

VisFactor 使用从 Factor-Referenced Cognitive Test (FRCT) 改编而来的 20 个以视觉为中心的子测试，评估多模态大语言模型的基础视觉认知能力。它聚焦于支撑高层视觉推理的基本能力，而非衡量单一下游任务的表现。

## 任务描述

- **任务类型**：包含二元判断和简短自由回答的视觉认知评估
- **输入**：1 至 4 张图像，与任务特定指令交错排列
- **输出**：一个 JSON 对象，包含布尔值、单词、数字、坐标对或字母形式的答案
- **领域**：可视化与空间处理、知觉闭合、视觉记忆及推理

## 主要特性

- 包含 3,046 行数据，对应 20 个 FRCT 子测试中的 808 个测试项
- 采用基于规则的变体和分组一致性检查，将平均随机猜测准确率降至约 2.9%
- 保留 VLMEvalKit 实现中的官方零样本提示及其图像顺序
- 覆盖隐藏图形识别、格式塔完形、视觉记忆、心理旋转、路径查找、折纸推理等相关能力

## 评估说明

- 使用官方 `VisFactor.tsv` 在 ModelScope 镜像中的 **test** 划分
- 提取最后一个 `{"answer": ...}` 对象，并应用官方针对不同类别的标准化规则
- 单个逻辑测试项可能包含多行数据，仅当所有行均正确时才计为正确
- 报告每个子测试在测试项级别的准确率；主得分是所涵盖子测试的未加权宏平均值
- 评分过程完全确定，无需依赖 LLM 评判器

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `visfactor` |
| **数据集ID** | [lmms-lab-encoder/visfactor](https://modelscope.cn/datasets/lmms-lab-encoder/visfactor/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2502.16435) |
| **标签** | `MultiModal`, `QA`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,046 |
| 提示词长度（平均） | 463.45 字符 |
| 提示词长度（最小/最大） | 188 / 932 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 6,048 |
| 每样本图像数 | 最小: 1, 最大: 4, 平均: 1.99 |
| 分辨率范围 | 100x100 - 668x911 |
| 格式 | jpeg |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "1a4f53fe",
      "content": [
        {
          "text": "Look at the two images:\n\nBelow is the first image, one simple shape:"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~2.6KB]"
        },
        {
          "text": "Below is the second image, a larger, complex pattern:"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~8.5KB]"
        },
        {
          "text": "Task: Decide whether the shape in the first image is hidden anywhere inside the second image. The shape will never be rotated, flipped, or resized. The shape will always be right-side-up and exactly the same size as in the first image.\n\nOutput: Respond with only one word: “TRUE” if it is present, “FALSE” if it is not, in JSON format as follows: {\"answer\": YOUR_ANSWER_HERE}."
        }
      ]
    }
  ],
  "target": "T",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "category_id": "CF1",
    "category_name": "Hidden Figures Test",
    "eval_index": 0,
    "additional": ""
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets visfactor \
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
    datasets=['visfactor'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
