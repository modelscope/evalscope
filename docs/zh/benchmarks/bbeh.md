# BBEH

## 概述

BIG-Bench Extra Hard（BBEH）将 BIG-Bench Hard 中的 23 个任务逐一替换为考察相似推理能力、但难度大幅提升的新任务。在 BBH 上已接近满分的前沿模型在 BBEH 上的得分要低得多，因此它能重新拉开不同模型在通用推理能力上的差距。

## 任务描述

- **任务类型**：通用推理（自由作答与多项选择）
- **输入**：任务相关的问题，后接官方的答案格式说明
- **输出**：逐步推理，并以 `The answer is: <answer>` 结尾
- **领域**：逻辑、时间、空间、因果、算术、语言与常识推理，以及幽默与讽刺理解

## 主要特点

- 共 23 个任务、4,520 个样本（每个任务 200 个，Disambiguation QA 为 120 个）
- BBEH Mini：官方的 460 个样本子集（每个任务 20 个），通过 `extra_params={'mini': True}` 选择
- 答案的提取与匹配使用 google-deepmind/bbeh 官方的 `evaluate.py`
- 输入较长：提示词平均约 4,300 个字符，最长达 32,000 个字符（Shuffled Objects 和 Zebra Puzzles 平均超过 11,000 个字符）

## 评估说明

- 0-shot，使用 BBEH 论文中的提示词后缀。
- 每个任务的 `acc` 使用官方的模糊匹配（不区分大小写、`(a)` 与 `a` 等价、数值相等、去除 LaTeX `\boxed{}` 及标记）。
- 整体 `acc` 为所有样本的微平均，即 BBEH 排行榜中的 "Micro Avg"。
- 报告中额外给出论文的主要指标，即 `harmonic_mean` 聚合的 `accuracy`：对各任务准确率求调和平均，并按论文做法给每个准确率加 1 个百分点以处理零分。
- 请为模型设置足够大的 `max_tokens`；论文指出，低于随机水平的得分多是因为回答在给出最终答案前被截断。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bbeh` |
| **数据集ID** | [evalscope/bbeh](https://modelscope.cn/datasets/evalscope/bbeh/summary) |
| **论文** | [论文](https://arxiv.org/abs/2502.19187) |
| **标签** | `Reasoning` |
| **指标** | `accuracy` |
| **默认样本数（Shots）** | 0-shot |
| **评估分割** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,520 |
| 提示词长度（平均） | 4285.09 字符 |
| 提示词长度（最小/最大） | 663 / 32506 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `boardgame_qa` | 200 | 7193.37 | 2269 | 14301 |
| `boolean_expressions` | 200 | 3120.27 | 2352 | 4165 |
| `buggy_tables` | 200 | 5448.78 | 4033 | 7202 |
| `causal_understanding` | 200 | 1349.26 | 663 | 2619 |
| `disambiguation_qa` | 120 | 1193.55 | 799 | 2202 |
| `dyck_languages` | 200 | 3277.7 | 2167 | 10978 |
| `geometric_shapes` | 200 | 2836.78 | 1836 | 4897 |
| `hyperbaton` | 200 | 4999.72 | 2751 | 7367 |
| `linguini` | 200 | 1344.07 | 726 | 2138 |
| `movie_recommendation` | 200 | 1504.48 | 1144 | 2172 |
| `multistep_arithmetic` | 200 | 1423.19 | 1187 | 1689 |
| `nycc` | 200 | 1297.05 | 1074 | 1514 |
| `object_counting` | 200 | 9205.75 | 5899 | 12884 |
| `object_properties` | 200 | 5714.45 | 4397 | 7127 |
| `sarc_triples` | 200 | 1717.4 | 1524 | 2467 |
| `shuffled_objects` | 200 | 13838.18 | 10574 | 17484 |
| `spatial_reasoning` | 200 | 6764.89 | 1116 | 32506 |
| `sportqa` | 200 | 2169.74 | 1194 | 3493 |
| `temporal_sequence` | 200 | 4156.03 | 2831 | 5497 |
| `time_arithmetic` | 200 | 1569.47 | 1030 | 2402 |
| `web_of_lies` | 200 | 2959.66 | 1491 | 4320 |
| `word_sorting` | 200 | 2436.12 | 733 | 6989 |
| `zebra_puzzles` | 200 | 11800.46 | 5318 | 15625 |

## 样例示例

**子集**: `boardgame_qa`

```json
{
  "input": [
    {
      "id": "e18d4e3b",
      "content": "A few players are playing a boardgame. The current state of the game is as follows. The bee has a football with a radius of 15 inches. The chihuahua has a banana-strawberry smoothie. The chihuahua lost her keys. The cobra suspects the truthfu ... [TRUNCATED 9092 chars] ... rkup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\"."
    }
  ],
  "target": "disproved",
  "id": 0,
  "group_id": 0,
  "subset_key": "boardgame_qa",
  "metadata": {
    "mini": false
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}

Think step by step, and when you provide the final answer, please use the prefix "The answer is:" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: "The answer is: 42" or "The answer is: yes". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, "The answer is: (a)".
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `mini` | `bool` | `False` | 评估 BBEH Mini，即官方的 460 个样本子集（每个任务 20 个）。 |

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bbeh \
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
    datasets=['bbeh'],
    dataset_args={
        'bbeh': {
            # subset_list: ['boardgame_qa', 'boolean_expressions', 'buggy_tables']  # 可选，用于评估特定子集
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
