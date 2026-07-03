# BabyVision


## 概述

BabyVision 是一个视觉感知基准测试，通过受婴儿及幼儿视觉发展启发的任务，评估多模态大语言模型的基础视觉能力。该基准聚焦于细粒度辨别、空间感知、视觉模式识别和视觉追踪能力。

## 任务描述

- **任务类型**：视觉感知（选择题 + 填空题）
- **输入**：图像 + 问题
- **输出**：选项字母或自由格式的简短答案
- **领域**：细粒度辨别、空间感知、视觉模式识别、视觉追踪

## 主要特点

- 包含 388 个测试样本，覆盖 4 大视觉能力类别和 22 个子类型
- 两种答案类型：选择题（135 个样本）和填空题（253 个样本）
- 子类型包括：找不同、找相同、数簇、迷宫、立方体展开图、图案补全、折纸、旋转图案等
- 侧重于低层次视觉感知，而非高层次推理或知识
- 提供思维链（Chain-of-Thought, CoT）参考用于分析

## 评估说明

- 默认评估使用 **train** 切分（388 个样本，单切分数据集）
- 主要指标：通过 LLM-as-judge 计算的 **准确率（Accuracy）**
- 子集按 `type` 字段组织（4 个类别）
- LLM 评判器统一评估选择题和填空题两种答案类型
- 需要配置 `judge_model_args` 以启用 LLM 评判器

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `baby_vision` |
| **数据集ID** | [evalscope/BabyVision](https://modelscope.cn/datasets/evalscope/BabyVision/summary) |
| **论文** | N/A |
| **标签** | `MultiModal`, `QA`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估切分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 388 |
| 提示词长度（平均） | 167.37 字符 |
| 提示词长度（最小/最大） | 33 / 450 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `Fine-grained Discrimination` | 163 | 152.09 | 33 | 450 |
| `Spatial Perception` | 91 | 157.18 | 73 | 370 |
| `Visual Pattern Recognition` | 51 | 178.92 | 94 | 319 |
| `Visual Tracking` | 83 | 201.45 | 97 | 389 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 388 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 174x144 - 2378x1448 |
| 格式 | jpeg, png, webp |


## 样例示例

**子集**: `Fine-grained Discrimination`

```json
{
  "input": [
    {
      "id": "8b83904b",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~77.0KB]"
        },
        {
          "text": "The image shows a total of 49 tiger patterns arranged in 7 rows and 7 columns. One of them is different from the others. Which row and column is it in? The answer format is (x,y). (For example, the answer for the 2nd row and 3rd column is (2,3))."
        }
      ]
    }
  ],
  "target": "(4,7)",
  "id": 0,
  "group_id": 0,
  "subset_key": "Fine-grained Discrimination",
  "metadata": {
    "taskId": 445,
    "type": "Fine-grained Discrimination",
    "subtype": "Find the different",
    "ansType": "blank",
    "coT": "The image shows 49 tiger patterns arranged in 7 rows and 7 columns.\nNow, we need to find the coordinates of the one tiger pattern that is different from the other 48.\nIt can be observed that the tiger in the fourth row and seventh column has no ears (the ears are located in the upper right corner of each tiger pattern), while the other 48 tigers have ears.\nTherefore, the correct answer is (4,7)."
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 通过命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets baby_vision \
    --limit 10  # 正式评估时请删除此行
```

### 通过 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['baby_vision'],
    dataset_args={
        'baby_vision': {
            # subset_list: ['Fine-grained Discrimination', 'Spatial Perception', 'Visual Pattern Recognition']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```