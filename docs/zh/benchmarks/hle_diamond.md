# Humanity's-Last-Exam-Diamond


## 概述

Humanity's Last Exam Diamond (HLE-Diamond) 是 Humanity's Last Exam 官方精选的包含 1,000 道题目的子集。它在保留原始专家级题目格式的同时，将推理和知识部分进行了独立划分。

## 任务描述

- **任务类型**: 专家级问答
- **输入**: 问题及可选的图片
- **输出**: 带有解释和置信度分数的答案
- **领域**: 广泛的学术和专业学科，分为推理和知识两部分

## 主要特点

- 1,000 道题目：500 道推理题和 500 道知识题
- 使用原始的 HLE 答案类型和多模态题目
- 报告整体准确率以及独立的推理/知识分区结果
- 保留原始题目元数据，包括学科类别和解题依据

## 评估说明

- 默认评估使用 **test** 分割集
- 主要指标：**Accuracy**（准确率），采用与 HLE 相同的 JSON C/I LLM 评判契约
- 响应格式包括：Explanation（解释）、Answer（答案）和 Confidence（置信度，0-100%）
- 对于纯文本模型，请将 `extra_params["include_multi_modal"]` 设置为 `False`
- 访问 Hugging Face 数据集需要接受数据集条款并提供访问令牌；同时也支持经过身份验证或本地数据集路径


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `hle_diamond` |
| **数据集ID** | [cais/hle-diamond](https://modelscope.cn/datasets/cais/hle-diamond/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `QA`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割集** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,000 |
| 提示词长度（平均值） | 890.94 字符 |
| 提示词长度（最小/最大值） | 232 / 18201 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `reasoning` | 500 | 1065.99 | 263 | 10668 |
| `knowledge` | 500 | 715.89 | 232 | 18201 |

**图片统计信息：**

| 指标 | 值 |
|--------|-------|
| 图片总数 | 272 |
| 每样本图片数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 329x12 - 14950x2780 |
| 格式 | gif, jpeg, png, webp |


## 样例示例

*暂无样例示例。*

## 提示模板

**提示模板:**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `include_multi_modal` | `bool` | `True` | 在评估期间是否包含多模态（图片）题目。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets hle_diamond \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
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
    datasets=['hle_diamond'],
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    dataset_args={
        'hle_diamond': {
            # subset_list: ['reasoning', 'knowledge']  # 可选，用于评估特定子集
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
