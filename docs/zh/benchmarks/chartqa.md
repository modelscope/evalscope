# ChartQA


## 概述

ChartQA 是一个用于评估模型在图表和数据可视化上问答能力的基准测试。它考察模型对各类图表（包括柱状图、折线图和饼图）的视觉推理与逻辑理解能力。

## 任务描述

- **任务类型**：图表问答（Chart Question Answering）
- **输入**：图表图像 + 自然语言问题
- **输出**：单个单词或数值答案
- **领域**：数据可视化、视觉推理、数值推理

## 主要特点

- 涵盖多种图表类型（柱状图、折线图、饼图、散点图）
- 包含人工编写和自动生成的测试问题
- 要求理解图表结构和数据关系
- 同时测试视觉信息提取与逻辑推理能力
- 问题难度从简单数据查询到复杂推理不等

## 评估说明

- 默认使用 **test** 数据划分，包含两个子集：
  - `human_test`：人工编写的问题
  - `augmented_test`：自动生成的问题
- 主要指标：**Relaxed Accuracy**（允许答案存在微小差异）
- 答案格式应为 "ANSWER: [ANSWER]"
- 数值答案允许存在舍入误差范围内的差异


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `chartqa` |
| **数据集ID** | [lmms-lab/ChartQA](https://modelscope.cn/datasets/lmms-lab/ChartQA/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `relaxed_acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,500 |
| 提示词长度（平均） | 224.33 字符 |
| 提示词长度（最小/最大） | 178 / 352 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `human_test` | 1,250 | 220.17 | 178 | 352 |
| `augmented_test` | 1,250 | 228.49 | 186 | 293 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 2,500 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 184x326 - 800x1796 |
| 格式 | png |


## 样例示例

**子集**: `human_test`

```json
{
  "input": [
    {
      "id": "c75439e0",
      "content": [
        {
          "text": "\nHow many food item is shown in the bar graph?\n\nThe last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the a single word answer or number to the problem.\n"
        },
        {
          "image": "[BASE64_IMAGE: png, ~42.9KB]"
        }
      ]
    }
  ],
  "target": "14",
  "id": 0,
  "group_id": 0,
  "subset_key": "human_test"
}
```

## 提示模板

**提示模板：**
```text

{question}

The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the a single word answer or number to the problem.

```


## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets chartqa \
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
    datasets=['chartqa'],
    dataset_args={
        'chartqa': {
            # subset_list: ['human_test', 'augmented_test']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```