# POPE


## 概述

POPE（Polling-based Object Probing Evaluation）是一个专门用于评估大视觉语言模型（LVLMs）中物体幻觉现象的基准测试。它通过是非题（yes/no questions）来检验模型能否准确识别图像中存在的物体。

## 任务描述

- **任务类型**：物体幻觉检测（是非问答）
- **输入**：图像 + 问题 “图像中是否有 [物体]？”
- **输出**：YES 或 NO
- **重点**：衡量准确率与幻觉率

## 核心特性

- 三种采样策略：随机（random）、热门（popular）、对抗（adversarial）
- 测试模型对不存在物体的错误肯定（即幻觉）
- 基于 MSCOCO 图像构建
- 采用简单的是非题格式，便于客观评估
- 衡量模型回答与视觉内容的一致性

## 评估说明

- 默认配置使用 **0-shot** 评估
- 五个指标：准确率（accuracy）、精确率（precision）、召回率（recall）、F1 分数（F1 score）、yes_ratio
- F1 分数为主要聚合指标
- 包含三个子集：`popular`、`adversarial`、`random`
- `popular` 和 `adversarial` 子集更具挑战性
- yes_ratio 反映模型倾向于回答 “yes” 的程度


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `pope` |
| **数据集ID** | [lmms-lab/POPE](https://modelscope.cn/datasets/lmms-lab/POPE/summary) |
| **论文** | N/A |
| **标签** | `Hallucination`, `MultiModal`, `Yes/No` |
| **指标** | `accuracy`, `precision`, `recall`, `f1_score`, `yes_ratio` |
| **默认示例数量** | 0-shot |
| **评估划分** | `N/A` |
| **聚合方式** | `f1` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 9,000 |
| 提示词长度（平均） | 79.4 字符 |
| 提示词长度（最小/最大） | 75 / 87 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `popular` | 3,000 | 79.27 | 75 | 87 |
| `adversarial` | 3,000 | 79.36 | 75 | 87 |
| `random` | 3,000 | 79.59 | 75 | 87 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 9,000 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 500x243 - 640x640 |
| 格式 | jpeg |


## 样例示例

**子集**: `popular`

```json
{
  "input": [
    {
      "id": "8847a5a3",
      "content": [
        {
          "text": "Is there a snowboard in the image?\nPlease answer YES or NO without an explanation."
        },
        {
          "image": "[BASE64_IMAGE: png, ~87.2KB]"
        }
      ]
    }
  ],
  "target": "YES",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "3000",
    "answer": "YES",
    "category": "popular",
    "question_id": "1"
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
Please answer YES or NO without an explanation.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets pope \
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
    datasets=['pope'],
    dataset_args={
        'pope': {
            # subset_list: ['popular', 'adversarial', 'random']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```