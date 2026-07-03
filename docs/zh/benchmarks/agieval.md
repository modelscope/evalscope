# AGIEval


## 概述

AGIEval 是一个以人类为中心的基准测试，旨在评估基础模型在人类认知与问题解决场景下的能力。该基准采用面向普通人类考生的官方、标准且权威的入学与资格考试题目，例如高考（GaoKao）、法学院入学考试（LSAT）、数学竞赛以及律师资格考试等。

## 任务描述

- **任务类型**：混合型（多项选择问答 + 开放式数学题）
- **输入**：标准化考试中的题目，可包含文章段落和选项
- **输出**：多项选择题的答案字母，或开放式题目的数值/数学答案
- **语言**：英语和中文

## 主要特点

- 包含21个子集，涵盖两种语言下的多种考试类型
- 英语多项选择题：LSAT（AR/LR/RC）、SAT（数学/英语）、AQuA-RAT、LogiQA、GaoKao-English
- 中文多项选择题：GaoKao（语文/地理/历史/生物/化学/物理/MathQA）、LogiQA-zh、JEC-QA
- 开放式数学题：MATH（英语）、GaoKao-MathCloze（中文）
- 多选子集：JEC-QA-KD、JEC-QA-CA、GaoKao-Physics
- 包含基于文章的阅读理解题目

## 评估说明

- 多项选择题子集采用精确字母匹配（单选或多选）
- 数学/填空类子集采用数学等价性检查
- 支持使用开发集（dev split）样例进行少样本（few-shot）评估
- 提示格式遵循官方 AGIEval 规范

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `agieval` |
| **数据集ID** | [opencompass/agieval](https://modelscope.cn/datasets/opencompass/agieval/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认样本数** | 0-shot |
| **评估划分** | `test` |
| **训练划分** | `dev` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 8,269 |
| 提示词长度（平均） | 673.58 字符 |
| 提示词长度（最小/最大） | 40 / 5316 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `aqua-rat` | 254 | 290.09 | 103 | 587 |
| `logiqa-en` | 651 | 911.89 | 248 | 1769 |
| `lsat-ar` | 230 | 946.36 | 635 | 1853 |
| `lsat-lr` | 510 | 1156.66 | 563 | 2348 |
| `lsat-rc` | 269 | 3652.86 | 2959 | 4825 |
| `sat-math` | 220 | 392.45 | 120 | 1201 |
| `sat-en` | 206 | 4618.28 | 3569 | 5316 |
| `sat-en-without-passage` | 206 | 435.91 | 169 | 937 |
| `gaokao-english` | 306 | 2025.44 | 517 | 4216 |
| `logiqa-zh` | 651 | 267.62 | 98 | 526 |
| `gaokao-chinese` | 246 | 988.09 | 152 | 2186 |
| `gaokao-geography` | 199 | 204.82 | 64 | 881 |
| `gaokao-history` | 235 | 141.48 | 67 | 314 |
| `gaokao-biology` | 210 | 203.98 | 75 | 685 |
| `gaokao-chemistry` | 207 | 348.37 | 58 | 1454 |
| `gaokao-physics` | 200 | 251.9 | 58 | 581 |
| `gaokao-mathqa` | 351 | 201.59 | 93 | 615 |
| `jec-qa-kd` | 1,000 | 170.43 | 54 | 454 |
| `jec-qa-ca` | 1,000 | 240.71 | 79 | 883 |
| `math` | 1,000 | 211.95 | 40 | 2186 |
| `gaokao-mathcloze` | 118 | 123.42 | 48 | 501 |

## 样例示例

**子集**: `aqua-rat`

```json
{
  "input": [
    {
      "id": "e28353e5",
      "content": "Q: A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. After how much more time will this car reach the base of the tower? Answer Choices: (A)5(√3 + 1) (B)6(√3 + √2) (C)7(√3 – 1) (D)8(√3 – 2) (E)None of these\nA: Among A through E, the answer is"
    }
  ],
  "choices": [
    "(A)5(√3 + 1)",
    "(B)6(√3 + √2)",
    "(C)7(√3 – 1)",
    "(D)8(√3 – 2)",
    "(E)None of these"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "subset": "aqua-rat",
    "has_passage": false
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
    --datasets agieval \
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
    datasets=['agieval'],
    dataset_args={
        'agieval': {
            # subset_list: ['aqua-rat', 'logiqa-en', 'lsat-ar']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```