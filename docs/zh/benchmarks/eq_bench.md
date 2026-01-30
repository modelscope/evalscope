# EQ-Bench

## 概述

EQ-Bench 是一个用于评估语言模型在情感智能任务上表现的基准测试。它通过评估模型对对话中角色可能产生的情感反应强度进行打分，来衡量其情感理解能力。

## 任务描述

- **任务类型**：情感智能评估
- **输入**：包含角色的对话场景
- **输出**：特定格式的情感强度评分
- **领域**：情感理解、社会认知

## 主要特点

- 测试模型预测对话中情感反应的能力
- 要求对多种可能情绪的强度进行评分
- 使用官方 EQ-Bench v2 评分算法
- 评分包含 Sigmoid 缩放以处理微小差异
- 引入调整常数，确保随机回答得分为 0

## 评估说明

- 默认评估使用 **validation**（验证）数据集划分
- 主要指标：**EQ-Bench Score**（0-100 分制，报告为 0-1 范围）
- 采用零样本（zero-shot）评估方式（不提供少样本示例）
- 模型响应必须包含特定 JSON-like 格式的情感评分
- 官方算法来源：[论文](https://arxiv.org/abs/2312.06281) | [官网](https://eqbench.com/)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `eq_bench` |
| **数据集 ID** | [evalscope/EQ-Bench](https://modelscope.cn/datasets/evalscope/EQ-Bench/summary) |
| **论文** | N/A |
| **标签** | `InstructionFollowing` |
| **指标** | `eq_bench_score` |
| **默认样本数** | 0-shot |
| **评估数据划分** | `validation` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 171 |
| 提示词长度（平均） | 1550.02 字符 |
| 提示词长度（最小/最大） | 922 / 3737 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "97129bc9",
      "content": "Your task is to predict the likely emotional responses of a character in this dialogue:\n\nRobert: Claudia, you've always been the idealist. But let's be practical for once, shall we?\nClaudia: Practicality, according to you, means bulldozing ev ... [TRUNCATED] ... ary:\n\nRemorseful: <score>\nIndifferent: <score>\nAffectionate: <score>\nAnnoyed: <score>\n\n\n[End of answer]\n\nRemember: zero is a valid score, meaning they are likely not feeling that emotion. You must score at least one emotion > 0.\n\nYour answer:"
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

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets eq_bench \
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
    datasets=['eq_bench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```