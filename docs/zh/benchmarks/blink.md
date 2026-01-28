# BLINK


## 概述

BLINK 是一个用于评估多模态大语言模型（MLLMs）核心视觉感知能力的基准测试。它将 14 个经典的计算机视觉任务转化为 3,807 道包含单张或多张图像及视觉提示的多项选择题。

## 任务描述

- **任务类型**：视觉感知多项选择问答
- **输入**：一张或多张图像 + 多项选择题
- **输出**：单个答案字母
- **领域**：视觉感知、对应关系、推理、检测

## 主要特点

- 覆盖 14 种多样化的视觉感知任务
- 支持单图和多图输入（最多 4 张图像）
- 测试基础的视觉理解能力
- 类别包括：艺术风格、计数、取证检测、智商测试、拼图、多视角推理、物体定位等
- 题目源自经典计算机视觉基准数据集

## 评估说明

- 默认使用 **val** 划分进行评估
- 主要指标：多项选择题的 **准确率（Accuracy）**
- 响应格式为 "ANSWER: [LETTER]"
- 结果可按 14 个不同的感知类别进行分析


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `blink` |
| **数据集ID** | [evalscope/BLINK](https://modelscope.cn/datasets/evalscope/BLINK/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `val` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,901 |
| 提示词长度（平均） | 577.53 字符 |
| 提示词长度（最小/最大） | 252 / 1125 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Art_Style` | 117 | 553 | 553 | 553 |
| `Counting` | 120 | 285.21 | 270 | 317 |
| `Forensic_Detection` | 132 | 480 | 480 | 480 |
| `Functional_Correspondence` | 130 | 1118.34 | 1113 | 1125 |
| `IQ_Test` | 150 | 884.6 | 548 | 922 |
| `Jigsaw` | 150 | 543 | 543 | 543 |
| `Multi-view_Reasoning` | 133 | 549 | 549 | 549 |
| `Object_Localization` | 122 | 531.86 | 527 | 548 |
| `Relative_Depth` | 124 | 359 | 359 | 359 |
| `Relative_Reflectance` | 134 | 498 | 498 | 498 |
| `Semantic_Correspondence` | 139 | 952 | 952 | 952 |
| `Spatial_Relation` | 143 | 263.97 | 252 | 282 |
| `Visual_Correspondence` | 172 | 587 | 587 | 587 |
| `Visual_Similarity` | 135 | 414 | 414 | 414 |

**图像统计数据：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 3,675 |
| 每样本图像数 | 最小: 1, 最大: 4, 平均: 1.93 |
| 分辨率范围 | 200x83 - 3072x4096 |
| 格式 | jpeg |


## 样例示例

**子集**: `Art_Style`

```json
{
  "input": [
    {
      "id": "a522940e",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format:\n'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B.\n\nSome most common art painting styles include Realism, Impressi ... [TRUNCATED] ...  of art paintings, use the first image as the reference image, and determine which one of the second or the third image shares the same style as the reference image?\nSelect from the following choices.\n(A) the second image\n(B) the third image\n"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~477.8KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~876.1KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~329.2KB]"
        }
      ]
    }
  ],
  "choices": [
    "the second image",
    "the third image"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format:
'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets blink \
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
    datasets=['blink'],
    dataset_args={
        'blink': {
            # subset_list: ['Art_Style', 'Counting', 'Forensic_Detection']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```