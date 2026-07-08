# ArXiv-Math

## 概述

ArXiv-Math 是一个包含 103 道研究级数学问题的基准测试，这些问题均摘自 arXiv 预印本论文。这些问题代表了前沿数学研究，用于评估语言模型在知识前沿对高级数学概念进行推理的能力。

## 任务描述

- **任务类型**：研究级数学问题求解  
- **输入**：来自 arXiv 论文的高级数学问题  
- **输出**：分步解答及最终答案  
- **难度**：研究级 / 研究生水平  

## 主要特点

- 103 道问题来源于 arXiv 预印本（2024 年 12 月至 2025 年 3 月）  
- 包含四个按月划分的子集：december、february、january、march  
- 覆盖多个数学领域：代数、组合数学、分析、几何、数论  
- 问题需要深厚的数学推理能力和领域专业知识  
- 代表当前数学研究难度的最前沿  

## 评估说明

- 默认配置使用 **0-shot** 评估  
- 答案应使用 `\boxed{}` 格式包裹，以便正确提取  
- 采用数值准确率指标，并结合符号等价性检查  
- 结果可按每月竞赛子集分别统计  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `arxivmath` |
| **数据集ID** | [evalscope/arxivmath](https://modelscope.cn/datasets/evalscope/arxivmath/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 103 |
| 提示词长度（平均） | 622.88 字符 |
| 提示词长度（最小/最大） | 224 / 1392 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `arxiv/december` | 17 | 720.88 | 256 | 1392 |
| `arxiv/february` | 32 | 573.78 | 269 | 1147 |
| `arxiv/january` | 23 | 711.17 | 325 | 1270 |
| `arxiv/march` | 31 | 554.32 | 224 | 1213 |

## 样例示例

**子集**: `arxiv/december`

```json
{
  "input": [
    {
      "id": "c7cbf85d",
      "content": "Problem:\nLet $k$ be a field, let $V$ be a $k$-vector space of dimension $d$, and let $G\\subseteq GL(V)$ be a finite group. Set $r:=\\dim_k (V^*)^G$ and assume $r\\ge 1$. Let $R:=k[V]^G$ be the invariant ring, and write its Hilbert quasi-polynom ... [TRUNCATED 71 chars] ... {d-2}+\\cdots+a_1(n)n+a_0(n),\n\\]\nwhere each $a_i(n)$ is a periodic function of $n$. Compute the sum of the indices $i\\in\\{0,1,\\dots,d-1\\}$ for which $a_i(n)$ is constant.\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
    }
  ],
  "target": "\\frac{r(2d-r-1)}{2}",
  "id": 0,
  "group_id": 0,
  "subset_key": "arxiv/december",
  "metadata": {
    "problem_idx": 1,
    "problem_type": [
      ""
    ],
    "source": 2512.00811
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
    --datasets arxivmath \
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
    datasets=['arxivmath'],
    dataset_args={
        'arxivmath': {
            # subset_list: ['arxiv/december', 'arxiv/february', 'arxiv/january']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
