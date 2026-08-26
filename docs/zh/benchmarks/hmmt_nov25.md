# HMMT-Nov-2025


## 概述

HMMT November 2025（MathArena）是一个具有挑战性的评估基准，源自2025年11月举办的哈佛-麻省理工数学竞赛（Harvard-MIT Mathematics Tournament, HMMT），这是全球最负盛名且难度最高的高中数学竞赛之一。该基准与 HMMT February 2025（`hmmt25`）是不同的竞赛。

## 任务描述

- **任务类型**：竞赛数学问题求解
- **输入**：HMMT 级别的数学问题
- **输出**：包含逐步推理的答案
- **领域**：代数、组合数学、几何和数论

## 主要特点

- 包含30道来自 HMMT November 2025 竞赛的题目
- 数据源自 MathArena 的 `hmmt_nov_2025` 数据集，并在 ModelScope 上镜像
- 题目难度极高，属于竞赛级别
- 考察高级数学推理能力
- 代表顶尖高中数学竞赛难度

## 评估说明

- 默认配置从 ModelScope 加载 `evalscope/hmmt_nov_2025` 数据集，并评估 `train` 分割
- 默认配置使用 **0-shot** 评估方式
- 答案应使用 `\boxed{}` 格式包裹，以便正确提取
- 数值准确性通过数学等价性检查进行验证，支持整数、分数、小数和符号表达式
- 无需额外的运行时依赖项

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `hmmt_nov25` |
| **数据集ID** | [evalscope/hmmt_nov_2025](https://modelscope.cn/datasets/evalscope/hmmt_nov_2025/summary) |
| **论文** | 无 |
| **标签** | `Math`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 30 |
| 提示词长度（平均） | 403.3 字符 |
| 提示词长度（最小/最大） | 203 / 759 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "e53f11c5",
      "content": "Problem:\nLet $ABCD$ be a rectangle. Let $X$ and $Y$ be points on segments $\\overlien{BC}$ and $\\overline{AD}$, respectively, such that $\\angle AXY = \\angle XYC = 90^\\circ$. Given that $AX : XY : YC = 1 : 2 : 1$ and $AB = 1$, compute $BC$.\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
    }
  ],
  "target": "3",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "problem_idx": 1,
    "problem_type": null
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
    --datasets hmmt_nov25 \
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
    datasets=['hmmt_nov25'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
