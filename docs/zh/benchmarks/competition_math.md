# Competition-MATH

## 概述

Competition-MATH 是一个包含 12,500 道高难度竞赛数学题目的综合性基准测试数据集，题目来源于 AMC、AIME 等知名数学竞赛。该基准旨在评估语言模型在高级数学推理方面的能力。

## 任务描述

- **任务类型**：竞赛数学问题求解  
- **输入**：来自数学竞赛的数学问题  
- **输出**：包含逐步推导过程的解答，并将最终答案置于 `\boxed{}` 中  
- **难度等级**：Level 1（最简单）到 Level 5（最难）

## 主要特点

- 包含 12,500 道数学竞赛题目  
- 提供五个难度等级，支持全面评估  
- 涵盖主题：代数、计数与概率、几何、中级代数、数论、预代数、微积分预备  
- 每道题目均附有人工撰写的解答  
- 专为评估高级数学推理能力而设计  

## 评估说明

- 默认配置使用 **4-shot** 示例并结合 Chain-of-Thought（思维链）提示  
- 从 `\boxed{}` 格式中提取答案  
- 使用数值等价性检查进行答案比对  
- 支持按难度等级和问题类型分析结果  
- 使用 math_parser 进行鲁棒的答案提取  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `competition_math` |
| **数据集ID** | [evalscope/competition_math](https://modelscope.cn/datasets/evalscope/competition_math/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数量** | 4-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 5,000 |
| 提示词长度（平均） | 1019.03 字符 |
| 提示词长度（最小/最大） | 675 / 3991 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `Level 1` | 437 | 947.84 | 842 | 2757 |
| `Level 2` | 894 | 816.87 | 682 | 2328 |
| `Level 3` | 1,131 | 822.14 | 675 | 2470 |
| `Level 4` | 1,214 | 949.04 | 768 | 2871 |
| `Level 5` | 1,324 | 1411.41 | 1187 | 3991 |

## 样例示例

**子集**: `Level 1`

```json
{
  "input": [
    {
      "id": "3848cf6a",
      "content": "Here are some examples of how to solve similar problems:\n\nProblem:\nWhen Joyce counts the pennies in her bank by fives, she has one left over. When she counts them by threes, there are two left over. What is the least possible number of pennie ... [TRUNCATED] ... of 12, how many eggs will be left over if all cartons are sold?\nSolution:\n4\nProblem:\nHow many of the six integers 1 through 6 are divisors of the four-digit number 1452?\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
    }
  ],
  "target": "5",
  "id": 0,
  "group_id": 0,
  "subset_key": "Level 1",
  "metadata": {
    "reasoning": "All numbers are divisible by $1$. The last two digits, $52$, form a multiple of 4, so the number is divisible by $4$, and thus $2$. $1+4+5+2=12$, which is a multiple of $3$, so $1452$ is divisible by $3$.  Since it is divisible by $2$ and $3$, it is divisible by $6$.  But it is not divisible by $5$ as it does not end in $5$ or $0$.  So the total is $\\boxed{5}$.",
    "type": "Number Theory"
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.

```

<details>
<summary>少样本（Few-shot）模板</summary>

```text
Here are some examples of how to solve similar problems:

{fewshot}
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.

```

</details>

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets competition_math \
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
    datasets=['competition_math'],
    dataset_args={
        'competition_math': {
            # subset_list: ['Level 1', 'Level 2', 'Level 3']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```