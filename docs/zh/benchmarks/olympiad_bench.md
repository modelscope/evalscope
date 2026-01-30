# OlympiadBench

## 概述

OlympiadBench 是一个奥林匹克级别的双语多模态科学基准测试，包含 8,476 道来自数学和物理竞赛的问题（包括中国高考），用于严格评估高级科学推理能力。

## 任务描述

- **任务类型**：奥林匹克级别数学/物理问题求解
- **输入**：问题文本（可选附带最多 9 张图片）
- **输出**：数学答案或证明
- **领域**：数学、物理（双语：英文和中文）

## 主要特点

- 包含 8,476 道奥林匹克级别问题
- 支持双语（英文和中文）
- 覆盖数学与物理两个学科
- 子集命名规则：
  - `OE`：开放性问题（Open-Ended）
  - `TP`：定理证明问题（Theorem Proving）
  - `MM`：多模态（含图片）
  - `TO`：纯文本（Text-Only）
  - `CEE`：中国高考（Chinese Entrance Exam）
  - `COMP`：综合竞赛题（Comprehensive competition problems）

## 评估说明

- 默认使用 **train** 数据划分进行评估
- 主要指标：基于数学判断的 **准确率（Accuracy）**
- 答案需使用 `\boxed{}` 格式
- **注意**：目前 `TP`（定理证明）子集无法自动评估
- 支持对近似答案设置数值精度/误差阈值

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `olympiad_bench` |
| **数据集 ID** | [AI-ModelScope/OlympiadBench](https://modelscope.cn/datasets/AI-ModelScope/OlympiadBench/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认提示方式** | 0-shot |
| **评估划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 8,476 |
| 提示词长度（平均） | 484.46 字符 |
| 提示词长度（最小/最大） | 129 / 5032 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `OE_MM_maths_en_COMP` | 150 | 1118.59 | 575 | 3943 |
| `OE_MM_maths_zh_CEE` | 1,910 | 343.89 | 182 | 1566 |
| `OE_MM_maths_zh_COMP` | 56 | 379.07 | 201 | 608 |
| `OE_MM_physics_en_COMP` | 456 | 863.57 | 556 | 2793 |
| `OE_MM_physics_zh_CEE` | 1,483 | 425.68 | 214 | 901 |
| `OE_TO_maths_en_COMP` | 674 | 825.93 | 545 | 4924 |
| `OE_TO_maths_zh_CEE` | 1,240 | 297.26 | 171 | 1098 |
| `OE_TO_maths_zh_COMP` | 408 | 322.72 | 173 | 1583 |
| `OE_TO_physics_en_COMP` | 236 | 949.1 | 548 | 3112 |
| `OE_TO_physics_zh_CEE` | 115 | 329.56 | 203 | 503 |
| `TP_MM_maths_en_COMP` | 62 | 2044.19 | 475 | 4617 |
| `TP_MM_maths_zh_CEE` | 652 | 241.05 | 165 | 674 |
| `TP_MM_maths_zh_COMP` | 81 | 304.7 | 206 | 512 |
| `TP_MM_physics_en_COMP` | 19 | 677.79 | 432 | 1602 |
| `TP_TO_maths_en_COMP` | 503 | 933.44 | 449 | 5032 |
| `TP_TO_maths_zh_CEE` | 207 | 242.79 | 141 | 947 |
| `TP_TO_maths_zh_COMP` | 199 | 285.43 | 129 | 522 |
| `TP_TO_physics_en_COMP` | 25 | 740.48 | 475 | 1262 |

**图片统计数据：**

| 指标 | 值 |
|--------|-------|
| 图片总数 | 5,875 |
| 每样本图片数 | 最小: 1，最大: 9，平均: 1.21 |
| 分辨率范围 | 64x46 - 1765x1947 |
| 图片格式 | jpeg, png |

## 样例示例

**子集**: `OE_MM_maths_en_COMP`

```json
{
  "input": [
    {
      "id": "d7b44a52",
      "content": [
        {
          "text": "The following is an open-ended problem from an International Math competition. The answer of The problem should be a numerical value. Please calculate the answer according to the given requirements and the information provided. Please use LaT ... [TRUNCATED] ... c_{3}, \\ldots$ with $c_{i}<C$ for all $i$, Turbo can (after studying the sequence) ensure that there is some point on the circle that it will never visit or crawl across.\n\nPlease reason step by step, and put your final answer within \\boxed{}."
        },
        {
          "image": "[BASE64_IMAGE: jpg, ~27.4KB]"
        }
      ]
    }
  ],
  "target": "$\\frac{1}{2}$",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": 2231,
    "subfield": "Geometry",
    "context": null,
    "solution": [
      "The largest possible $C$ is $C=\\frac{1}{2}$.\n\nFor $0<C \\leqslant \\frac{1}{2}$, Turbo can simply choose an arbitrary point $P$ (different from its starting point) to avoid. When Turbo is at an arbitrary point $A$ different from $P$, the two ar ... [TRUNCATED] ... Note: Every sequence of the form $c_{i}=x$ if $i$ is odd, and $c_{i}=y$ if $i$ is even, where $0<x, y<C$, such that $x+y \\geqslant 1$, and $x \\neq y$ satisfies the conditions with the same argument. There might be even more possible examples.",
      "To show that $C\\le \\frac12$\n\nWe consider the following related problem:\n\nWe assume instead that the snail Chet is moving left and right on the real line. Find the size $M$ of the smallest (closed) interval, that we cannot force Chet out of, u ... [TRUNCATED] ... n, 1-\\varepsilon]$. Indeed the absolute value of the final position is at least $1-\\frac{5}{6} \\varepsilon$. This contradicts the assumption, that we cannot force Chet out of $[-1+\\varepsilon, 1-\\varepsilon]$. Hence $M \\geqslant 2$ as needed."
    ],
    "final_answer": [
      "$\\frac{1}{2}$"
    ],
    "is_multiple_answer": false,
    "unit": null,
    "answer_type": "Numerical",
    "question_type": "Open-ended",
    "language": "English",
    "subject": "Math",
    "error": null
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
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
    --datasets olympiad_bench \
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
    datasets=['olympiad_bench'],
    dataset_args={
        'olympiad_bench': {
            # subset_list: ['OE_MM_maths_en_COMP', 'OE_MM_maths_zh_CEE', 'OE_MM_maths_zh_COMP']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```