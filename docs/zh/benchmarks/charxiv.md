# CharXiv


## 概述

CharXiv 是 NeurIPS 2024 提出的一个全面的图表理解基准测试，用于评估多模态大语言模型在来自 arXiv 论文的真实科学图表上的表现。该基准同时考察模型对图表元素的低层次感知能力（描述性任务）和对图表数据的高层次推理能力。

## 任务描述

- **任务类型**：图表理解（描述性 + 推理性）
- **输入**：科学图表图像 + 问题
- **输出**：自由格式文本答案
- **领域**：cs、physics、math、eess、q-bio、q-fin、stat、econ

## 主要特点

- 包含来自 8 个学科 arXiv 论文的 2,323 张真实科学图表
- 两种问题类型：
  - **描述性**（每张图表 4 个）：基础元素识别（标题、坐标轴、图例、趋势等）
  - **推理型**（每张图表 1 个）：需要综合数据进行高阶推理
- 19 种描述性问题模板，涵盖信息提取、枚举、模式识别、计数和组合性任务
- 4 种推理答案类型：图表内文本、通用文本、图表内数值、通用数值
- 验证集（1,000 张图表）和测试集（1,323 张图表）
- 依据官方 CharXiv 评分协议，通过 LLM 作为评判者进行评估

## 评估说明

- 默认评估使用 **验证集**（1,000 张图表，5,000 个问题）
- 每张图表生成 5 个样本：4 个描述性 + 1 个推理型
- 主要指标：通过 LLM-as-judge 计算的 **准确率（Accuracy）**
- 子集划分：`descriptive` 和 `reasoning`（也可按类别细分）
- 需配置 `judge_model_args` 以指定 LLM 评判模型
- [论文](https://arxiv.org/abs/2406.18521) | [GitHub](https://github.com/princeton-nlp/CharXiv)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `charxiv` |
| **数据集ID** | [princeton-nlp/CharXiv](https://modelscope.cn/datasets/princeton-nlp/CharXiv/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2406.18521) |
| **标签** | `MultiModal`, `QA`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 5,000 |
| 提示词长度（平均） | 276.24 字符 |
| 提示词长度（最小/最大） | 80 / 687 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `descriptive` | 4,000 | 261.51 | 156 | 432 |
| `reasoning` | 1,000 | 335.14 | 80 | 687 |

**图像统计数据：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 5,000 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 1023x139 - 1024x1024 |
| 格式 | jpeg |


## 样例示例

**子集**: `descriptive`

```json
{
  "input": [
    {
      "id": "44ff5b8a",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~70.0KB]"
        },
        {
          "text": "For the current plot, what is the spatially highest labeled tick on the y-axis?\n* Your final answer should be the tick value on the y-axis that is explicitly written. Ignore units or scales that are written separately from the tick."
        }
      ]
    }
  ],
  "target": "60",
  "id": 0,
  "group_id": 0,
  "subset_key": "descriptive",
  "metadata": {
    "question_type": "descriptive",
    "question_id": 7,
    "category": "cs",
    "original_id": "2004.10956"
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
    --datasets charxiv \
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
    datasets=['charxiv'],
    dataset_args={
        'charxiv': {
            # subset_list: ['descriptive', 'reasoning']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
