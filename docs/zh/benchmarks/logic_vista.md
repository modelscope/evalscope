# LogicVista


## 概述

LogicVista 评估多模态大语言模型在视觉场景下的基础逻辑推理能力。每个题目均为多项选择题，其选项直接绘制在图像中（如图表、谜题、序列、图形等），因此模型必须读取并推理这些视觉选项，而非文本选项。

## 任务描述

- **任务类型**：视觉逻辑推理（多项选择）
- **输入**：包含标注选项的图像 + 问题文本
- **输出**：所选选项的标签
- **领域**：抽象与图示逻辑推理

## 主要特点

- 包含 448 道人工标注的视觉多项选择题，来源于各类能力与推理测试
- 按五种推理技能划分子集：归纳、演绎、数值、空间和机械推理
- 答案选项位于图像内，每道题的标签范围不同（通常为 A-D 或 A-E，最多至 A-I）
- 少量题目允许多选（例如“哪两个方案能完成该图”），其标准答案为一组标签

## 评估说明

- 默认使用 **test** 划分进行评估，报告整体及各推理技能子集的 **准确率（Accuracy）**
- 使用思维链（Chain-of-thought）提示；从回复末尾的 `ANSWER:` 行提取标签；对于多选题，将预测答案与标准答案作为无序集合进行比较，符合官方评分规则
- 设置较宽松的 `max_tokens`：若回复在 `ANSWER:` 行前被截断，则从回复最后一个大写字母猜测答案，这是一种宽容的处理方式
- 在发布的 448 道题目中，有 2 道无法按原样评分：`v1_382` 既无问题也无答案，被跳过；`v1_20` 的选项使用数字标注，而基于字母的答案解析器无法匹配——参考实现采用相同处理方式

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `logic_vista` |
| **数据集ID** | [evalscope/LogicVista](https://modelscope.cn/datasets/evalscope/LogicVista/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2407.04973) |
| **标签** | `MCQ`, `MultiModal`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 447 |
| 提示词长度（平均） | 529.27 字符 |
| 提示词长度（最小/最大） | 397 / 900 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `inductive` | 107 | 463.01 | 406 | 645 |
| `deductive` | 93 | 577.96 | 426 | 790 |
| `numerical` | 95 | 582.38 | 460 | 811 |
| `spatial` | 78 | 450.85 | 397 | 747 |
| `mechanical` | 74 | 578.35 | 450 | 900 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 447 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 156x165 - 1328x1352 |
| 格式 | png |


## 样例示例

**子集**: `inductive`

```json
{
  "input": [
    {
      "id": "3a2564ac",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~46.7KB]"
        },
        {
          "text": "Answer the following multiple choice question. The answer options are shown in the image.\nThe last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is the label of the option you choose. If more than one option is correct, list all of their labels on that line. Think step by step before answering.\n\nWhat choice (A, B, C, or D) should be in place of the question mark that fits the pattern?"
        }
      ]
    }
  ],
  "choices": [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I"
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "subset_key": "inductive",
  "metadata": {
    "id": "v1_0"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The answer options are shown in the image.
The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is the label of the option you choose. If more than one option is correct, list all of their labels on that line. Think step by step before answering.

{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets logic_vista \
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
    datasets=['logic_vista'],
    dataset_args={
        'logic_vista': {
            # subset_list: ['inductive', 'deductive', 'numerical']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
