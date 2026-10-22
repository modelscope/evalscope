# EQ-Bench

## 概述

EQ-Bench 是一个用于评估语言模型在情感智能任务上的基准测试。它通过评估对话中角色可能的情绪反应强度来衡量预测情绪反应的能力。

## 任务描述

- **任务类型**: 情感智能评估
- **输入**: 包含角色的对话场景
- **输出**: 特定格式的情绪强度评分
- **领域**: 情感理解、社会认知

## 主要特点

- 测试在对话中预测情绪反应的能力
- 需要对多种可能的情绪进行强度评分
- 使用官方的 EQ-Bench v2 评分算法
- 评分包括对小差异的 Sigmoid 缩放
- 调整常数确保随机答案得分为 0

## 评估说明

- 默认评估使用 **验证** 分割
- 主要指标: **EQ-Bench 得分**，存储为官方得分除以 100 并显示在官方量表上
- 有效的每题得分可以是负数（存储范围：-1.9908 到 1）；在平均之前不会被截断
- 无法解析或无效的答案以及缺少 v2 全量表参考的样本将被排除在得分平均之外；指标计数是已评分样本的数量
- 使用零样本评估（无少量样本示例）
- 响应必须包含四个情绪评分，格式为 `Emotion: score`
- 该适配器不重现官方运行程序的格式失败重采样或可选修订模式
- 官方算法来自 [论文](https://arxiv.org/abs/2312.06281) | [主页](https://eqbench.com/)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `eq_bench` |
| **数据集ID** | [evalscope/EQ-Bench](https://modelscope.cn/datasets/evalscope/EQ-Bench/summary) |
| **论文** | N/A |
| **标签** | `InstructionFollowing` |
| **指标** | `eq_bench_score` |
| **默认示例数量** | 0-shot |
| **评估分割** | `validation` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 171 |
| 提示词长度（平均值） | 1550.02 字符 |
| 提示词长度（最小/最大值） | 922 / 3737 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "0e3345b4",
      "content": "Your task is to predict the likely emotional responses of a character in this dialogue:\n\nRobert: Claudia, you've always been the idealist. But let's be practical for once, shall we?\nClaudia: Practicality, according to you, means bulldozing ev ... [TRUNCATED 630 chars] ... ary:\n\nRemorseful: <score>\nIndifferent: <score>\nAffectionate: <score>\nAnnoyed: <score>\n\n\n[End of answer]\n\nRemember: zero is a valid score, meaning they are likely not feeling that emotion. You must score at least one emotion > 0.\n\nYour answer:"
    }
  ],
  "target": "{'emotion1': 'Remorseful', 'emotion2': 'Indifferent', 'emotion3': 'Affectionate', 'emotion4': 'Annoyed', 'emotion1_score': 2, 'emotion2_score': 3, 'emotion3_score': 0, 'emotion4_score': 5}",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "reference_answer": {
      "emotion1": "Remorseful",
      "emotion2": "Indifferent",
      "emotion3": "Affectionate",
      "emotion4": "Annoyed",
      "emotion1_score": 2,
      "emotion2_score": 3,
      "emotion3_score": 0,
      "emotion4_score": 5
    },
    "reference_answer_fullscale": {
      "emotion1": "Remorseful",
      "emotion2": "Indifferent",
      "emotion3": "Affectionate",
      "emotion4": "Annoyed",
      "emotion1_score": 0,
      "emotion2_score": "6",
      "emotion3_score": 0,
      "emotion4_score": "7"
    }
  }
}
```

*注：部分内容因显示需要被截断。*

## 提示模板

**提示模板:**
```text
{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets eq_bench \
    --limit 10  # Remove this line for formal evaluation
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['eq_bench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
