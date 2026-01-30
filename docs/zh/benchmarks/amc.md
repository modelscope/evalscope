# AMC


## 概述

AMC（美国数学竞赛）是一个基于2022-2024年AMC 10/12竞赛题目的基准测试。这些选择题考察高中阶段的数学解题能力，并作为晋级AIME竞赛的资格赛。

## 任务描述

- **任务类型**：竞赛数学（选择题）
- **输入**：AMC级别的数学问题
- **输出**：带逐步推理过程的正确答案
- **涵盖年份**：2022、2023、2024

## 主要特点

- 题目来自2022-2024年的AMC 10和AMC 12竞赛
- 五选一的多项选择题格式
- 考察主题：代数、几何、数论、组合数学
- 难度从基础到高阶不等
- 使用官方竞赛题目，配有已验证的解答

## 评估说明

- 默认配置采用 **0-shot** 评估
- 答案需用 `\boxed{}` 格式包裹以便正确提取
- 提供三个子集：`amc22`、`amc23`、`amc24`
- 题目包含原始URL供参考
- 元数据中提供解答用于验证


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `amc` |
| **数据集ID** | [evalscope/amc_22-24](https://modelscope.cn/datasets/evalscope/amc_22-24/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `N/A` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 134 |
| 提示词长度（平均） | 324.58 字符 |
| 提示词长度（最小/最大） | 98 / 1218 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `amc22` | 43 | 337.42 | 129 | 934 |
| `amc23` | 46 | 337.98 | 143 | 1218 |
| `amc24` | 45 | 298.62 | 98 | 882 |

## 样例示例

**子集**: `amc22`

```json
{
  "input": [
    {
      "id": "54851a8f",
      "content": "What is the value of\\[3+\\frac{1}{3+\\frac{1}{3+\\frac13}}?\\]\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "\\frac{109}{33}",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "year": 2022,
    "url": "https://artofproblemsolving.com/wiki/index.php/2022_AMC_12A_Problems/Problem_1",
    "solution": "We have\\begin{align*} 3+\\frac{1}{3+\\frac{1}{3+\\frac13}} &= 3+\\frac{1}{3+\\frac{1}{\\left(\\frac{10}{3}\\right)}} \\\\ &= 3+\\frac{1}{3+\\frac{3}{10}} \\\\ &= 3+\\frac{1}{\\left(\\frac{33}{10}\\right)} \\\\ &= 3+\\frac{10}{33} \\\\ &= \\boxed{\\textbf{(D)}\\ \\frac{109}{33}}. \\end{align*}"
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
    --datasets amc \
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
    datasets=['amc'],
    dataset_args={
        'amc': {
            # subset_list: ['amc22', 'amc23', 'amc24']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```