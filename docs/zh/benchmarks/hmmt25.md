# HMMT25

## 概述

HMMT February 2025（MathArena）是一个具有挑战性的评测基准，源自2025年2月举办的哈佛-麻省理工数学竞赛（Harvard-MIT Mathematics Tournament, HMMT），这是全球最负盛名且难度最高的高中数学竞赛之一。

## 任务描述

- **任务类型**：竞赛数学问题求解  
- **输入**：HMMT级别的数学问题  
- **输出**：包含逐步推理过程的答案  
- **难度**：高级高中竞赛水平  

## 主要特点

- 题目来自 HMMT 2025 年 2 月竞赛  
- 四大核心领域：代数（Algebra）、组合数学（Combinatorics）、几何（Geometry）、数论（Number Theory）  
- 高难度的竞赛级题目  
- 考察高级数学推理能力  
- 代表顶尖高中数学竞赛难度  

## 评测说明

- 默认配置使用 **0-shot** 评测  
- 答案应使用 `\boxed{}` 格式包裹，以便正确提取  
- 使用数值准确率（numeric accuracy）作为评估指标  
- 题目覆盖多个数学领域  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `hmmt25` |
| **数据集ID** | [evalscope/hmmt_feb_2025](https://modelscope.cn/datasets/evalscope/hmmt_feb_2025/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评测划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 30 |
| 提示词长度（平均） | 410.53 字符 |
| 提示词长度（最小/最大） | 173 / 1594 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "fb6fb4e8",
      "content": "Problem:\nCompute the sum of the positive divisors (including $1$) of $9!$ that have units digit $1$.\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
    }
  ],
  "target": "103",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "problem_idx": 1,
    "problem_type": [
      "Number Theory"
    ]
  }
}
```

## 提示模板

**提示模板：**
```text
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.

```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets hmmt25 \
    --limit 10  # 正式评测时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['hmmt25'],
    limit=10,  # 正式评测时请删除此行
)

run_task(task_cfg=task_cfg)
```