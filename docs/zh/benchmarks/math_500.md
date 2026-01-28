# MATH-500

## 概述

MATH-500 是从 MATH 基准测试中精心挑选出的 500 道题目组成的子集，旨在评估语言模型的数学推理能力。该数据集涵盖五个难度等级，涉及代数、几何、数论、微积分等多个数学主题。

## 任务描述

- **任务类型**：数学问题求解  
- **输入**：数学问题描述  
- **输出**：包含逐步推导过程和最终数值答案的解答  
- **难度等级**：Level 1（最简单）到 Level 5（最难）

## 主要特点

- 从完整的 MATH 数据集中精选出 500 道题目  
- 提供五个难度等级，支持细粒度评估  
- 题目覆盖代数、几何、数论、概率等多个领域  
- 每道题均附带参考解答  
- 在保证评估全面性的同时兼顾效率

## 评估说明

- 默认配置采用 **0-shot** 评估方式  
- 答案需使用 `\boxed{}` 格式包裹，以便正确提取  
- 使用数值等价性检查进行答案比对  
- 可按难度等级分别统计结果  
- 因规模适中，常用于数学推理能力的基准测试

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `math_500` |
| **数据集ID** | [AI-ModelScope/MATH-500](https://modelscope.cn/datasets/AI-ModelScope/MATH-500/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 500 |
| 提示词长度（平均） | 266.89 字符 |
| 提示词长度（最小/最大） | 91 / 1804 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Level 1` | 43 | 193.19 | 100 | 571 |
| `Level 2` | 90 | 218.82 | 91 | 802 |
| `Level 3` | 105 | 236 | 93 | 688 |
| `Level 4` | 128 | 277.1 | 93 | 1771 |
| `Level 5` | 134 | 337.28 | 118 | 1804 |

## 样例示例

**子集**: `Level 1`

```json
{
  "input": [
    {
      "id": "b5d90091",
      "content": "Suppose $\\sin D = 0.7$ in the diagram below. What is $DE$? [asy]\npair D,E,F;\nF = (0,0);\nD = (sqrt(51),7);\nE = (0,7);\ndraw(D--E--F--D);\ndraw(rightanglemark(D,E,F,15));\nlabel(\"$D$\",D,NE);\nlabel(\"$E$\",E,NW);\nlabel(\"$F$\",F,SW);\nlabel(\"$7$\",(E+F)/2,W);\n[/asy]\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "\\sqrt{51}",
  "id": 0,
  "group_id": 0,
  "subset_key": "Level 1",
  "metadata": {
    "question_id": "test/precalculus/1303.json",
    "solution": "The triangle is a right triangle, so $\\sin D = \\frac{EF}{DF}$. Then we have that $\\sin D = 0.7 = \\frac{7}{DF}$, so $DF = 10$.\n\nUsing the Pythagorean Theorem, we find that the length of $DE$ is $\\sqrt{DF^2 - EF^2},$ or $\\sqrt{100 - 49} = \\boxed{\\sqrt{51}}$."
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets math_500 \
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
    datasets=['math_500'],
    dataset_args={
        'math_500': {
            # subset_list: ['Level 1', 'Level 2', 'Level 3']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```