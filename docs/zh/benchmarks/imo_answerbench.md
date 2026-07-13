# IMO-AnswerBench

## 概述

IMO-AnswerBench 是一个包含 400 道高难度问题的基准测试集，题目均选自国际数学奥林匹克（IMO）备选题（Shortlists）。该基准涵盖四大数学领域，旨在评估语言模型在奥林匹克级别上的高级数学推理能力。

## 任务描述

- **任务类型**：奥林匹克数学问题求解
- **输入**：IMO 备选题中的数学问题
- **输出**：包含逐步推导过程及最终答案的完整解答
- **难度**：国际奥林匹克级别

## 主要特点

- 包含 400 道来自 IMO 备选题（2005–2024 年）的问题
- 覆盖四大领域：代数（Algebra）、组合数学（Combinatorics）、几何（Geometry）、数论（Number Theory）
- 子类别包括：运算（Operation）、不等式（Inequality）、数列（Sequence）、多项式（Polynomial）、函数方程（Functional Equation）等
- 答案形式多样，从简单整数到复杂的 LaTeX 表达式（如区间、集合、分数等）
- 代表当前数学问题求解基准中最高难度水平

## 评测说明

- 默认配置采用 **0-shot** 评测方式
- 答案需用 `\boxed{}` 包裹，以便正确提取
- 对复杂答案采用基于 LLM-as-judge 的数值等价性检查
- 结果可按类别（代数、组合、几何、数论）细分统计
- 许多答案涉及符号表达式，需进行数学等价性判断

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `imo_answerbench` |
| **数据集ID** | [evalscope/imo-answerbench](https://modelscope.cn/datasets/evalscope/imo-answerbench/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评测划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 400 |
| 提示词长度（平均） | 423.02 字符 |
| 提示词长度（最小/最大） | 145 / 1343 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Algebra` | 100 | 345.06 | 183 | 984 |
| `Combinatorics` | 100 | 641.53 | 257 | 1343 |
| `Geometry` | 100 | 375.44 | 196 | 703 |
| `Number theory` | 100 | 330.07 | 145 | 769 |

## 样例示例

**子集**: `Algebra`

```json
{
  "input": [
    {
      "id": "afe74111",
      "content": "Problem:\nFor a given positive integer $N$, Henry writes the quotient of $ab$ divided by $N+1$ on the board for each integer pair $(a,b)$ where $1\\le a,b\\le N$. Find all $N$ such that the sum of the $N^2$ numbers Henry wrote on the board is $\\frac{N^3-N^2+2}{4}$.\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
    }
  ],
  "target": "3",
  "id": 0,
  "group_id": 0,
  "subset_key": "Algebra",
  "metadata": {
    "problem_id": "imo-bench-algebra-001",
    "subcategory": "Operation",
    "source": "IMO Shortlist 2021"
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
    --datasets imo_answerbench \
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
    datasets=['imo_answerbench'],
    dataset_args={
        'imo_answerbench': {
            # subset_list: ['Algebra', 'Combinatorics', 'Geometry']  # 可选，用于评测特定子集
        }
    },
    limit=10,  # 正式评测时请删除此行
)

run_task(task_cfg=task_cfg)
```
