# POPE


## 概述

POPE（Polling-based Object Probing Evaluation）是一个专门用于评估大视觉语言模型（LVLMs）中物体幻觉现象的基准测试。它通过是非题（yes/no questions）来检验模型能否准确识别图像中存在的物体。

## 任务描述

- **任务类型**：物体幻觉检测（是非问答）
- **输入**：图像 + 问题 “图像中是否有 [物体]？”
- **输出**：YES 或 NO
- **重点**：衡量准确率与幻觉率

## 主要特点

- 三种采样策略：随机（random）、流行（popular）、对抗（adversarial）
- 测试模型对不存在物体做出肯定回答的倾向（即幻觉）
- 基于 MSCOCO 图像数据集
- 采用简单的是非题格式，便于客观评估
- 衡量模型回答与视觉内容之间的一致性

## 评估说明

- 默认配置使用 **0-shot** 评估
- 五个指标：准确率（accuracy）、精确率（precision）、召回率（recall）、F1 分数（F1 score）、肯定回答比例（yes_ratio）
- 准确率是主要指标；精确率、召回率、F1 和 yes_ratio 提供辅助诊断信息
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
| **指标** | `accuracy`, `precision`, `recall`, `f1`, `yes_ratio` |
| **默认示例数** | 0-shot |
| **评估划分** | `N/A` |
| **聚合方式** | `f1` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
{question}
请仅回答 YES 或 NO，无需解释。
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
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
