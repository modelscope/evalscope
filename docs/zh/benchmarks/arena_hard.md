# ArenaHard

## 概述

ArenaHard 是一个具有挑战性的基准测试，通过竞争性成对比较来评估语言模型。模型在需要推理、理解和生成能力的困难任务上与 GPT-4 基线进行对比评判。

## 任务描述

- **任务类型**：竞争性模型评估（竞技场风格）
- **输入**：具有挑战性的指令/问题
- **输出**：模型响应，与 GPT-4-0314 基线进行比较
- **评分方式**：基于 Elo 评分的成对对战结果

## 主要特性

- 包含 500 个具有挑战性的用户提示
- 采用两局对战系统（A vs B 和 B vs A）
- 使用 Elo 评分计算模型排名
- 测试推理能力、指令遵循能力和文本生成能力
- 与 Chatbot Arena 排名高度相关

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用 LLM 作为裁判（默认：gpt-4-1106-preview）
- 基线模型：gpt-4-0314 的输出
- 报告胜率和基于 Elo 的得分
- 注意：目前不支持风格控制的胜率计算

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `arena_hard` |
| **数据集ID** | [AI-ModelScope/arena-hard-auto-v0.1](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary) |
| **论文** | N/A |
| **标签** | `Arena`, `InstructionFollowing` |
| **指标** | `winrate` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `elo` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 500 |
| 提示词长度（平均） | 406.36 字符 |
| 提示词长度（最小/最大） | 29 / 9140 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "088243c7",
      "content": "Use ABC notation to write a melody in the style of a folk tune."
    }
  ],
  "target": "X:1\nT:Untitled Folk Tune\nM:4/4\nL:1/8\nK:G\n|:G2A2|B2A2|G2E2|D4|E2F2|G2F2|E2C2|B,4|\nA2B2|c2B2|A2F2|E4|D2E2|F2E2|D2B,2|C4:|",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "capability": "ABC Sequence Puzzles & Groups"
  }
}
```

## 提示模板

**提示模板：**
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
    --datasets arena_hard \
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
    datasets=['arena_hard'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```