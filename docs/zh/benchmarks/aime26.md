# AIME-2026


## 概述

AIME 2026（2026年美国数学邀请赛）是一个基于著名AIME竞赛题目的基准测试。AIME是美国最具挑战性的高中数学竞赛之一，该基准测试旨在评估高级数学推理与问题解决能力。

## 任务描述

- **任务类型**：竞赛数学问题求解
- **输入**：AIME级别的数学问题
- **输出**：整数答案（0-999），附带逐步推理过程
- **难度**：高级高中 / 大学低年级水平

## 主要特点

- 题目来自2026年AIME I和AIME II竞赛
- 答案始终为0到999之间的整数
- 要求具备创造性数学推理与解题能力
- 涉及主题：代数、几何、数论、组合数学、概率
- 代表顶尖高中数学竞赛难度

## 评估说明

- 默认配置采用 **0-shot** 评估方式
- 答案需用 `\boxed{}` 格式包裹以便正确提取
- 使用LLM-as-judge进行数学等价性检查

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `aime26` |
| **数据集ID** | [evalscope/aime26](https://modelscope.cn/datasets/evalscope/aime26/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 30 |
| 提示词长度（平均） | 517.7 字符 |
| 提示词长度（最小/最大） | 263 / 1026 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "a6d469bc",
      "content": "\nSolve the following math problem step by step. Put your answer inside \\boxed{}.\n\nPatrick started walking at a constant rate along a straight road from school to the park. One hour after Patrick left, Tanya started running along the same road ... [TRUNCATED 270 chars] ... ya ran, and all three arrived at the park at the same time. The distance from the school to the park is $\\frac{m}{n}$ miles, where $m$ and $n$ are relatively prime positive integers. Find $m + n$.\n\nRemember to put your answer inside \\boxed{}."
    }
  ],
  "target": "277",
  "id": 0,
  "group_id": 0
}
```

## 提示模板

**提示模板：**
```text

Solve the following math problem step by step. Put your answer inside \boxed{{}}.

{question}

Remember to put your answer inside \boxed{{}}.
```

## 使用方法

### 使用命令行接口（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets aime26 \
    --limit 10  # 正式评估时请删除此行
```

### 使用Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['aime26'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```