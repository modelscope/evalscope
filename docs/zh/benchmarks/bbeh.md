# BBEH

## 概述

BIG-Bench Extra Hard (BBEH) 通过 23 项任务评估高难度的通用推理能力。它基于 BIG-Bench Hard 构建，涵盖了类似范围的推理技能，但题目难度更高。

## 任务描述

- **任务类型**：通用推理（自由文本和多项选择题）
- **输入**：带有答案格式说明的推理问题
- **输出**：推理过程后跟随 `The answer is: <answer>`
- **领域**：逻辑、算术、语言、常识、时间与空间推理、幽默与讽刺

## 主要特点

- 完整基准测试：包含 4,520 个样本，涵盖 23 项任务（每项任务 200 个样本，Disambiguation QA 为 120 个样本）
- 可选的 BBEH Mini：包含 460 个样本，每项任务 20 个样本，用于更快速的评估
- 问题格式多样且输入较长，提示词长度可达约 32,500 个字符

## 评估说明

- 使用官方的零样本提示和答案匹配规则，包括其在标点符号方面的局限性。
- 报告每项任务的准确率、总体微平均准确率以及论文中的 BBEH Score。
- BBEH Score 是各项任务准确率加 1 个百分点后的调和平均值。其范围为 1% 到 101%，而非通常的 0% 到 100% 准确率范围。
- 为了进行整体比较，请评估所有 23 项任务。确保分配足够的输出 token：没有最终答案的回答可能会被标记为错误。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bbeh` |
| **数据集ID** | [evalscope/bbeh](https://modelscope.cn/datasets/evalscope/bbeh/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2502.19187) |
| **标签** | `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,520 |
| 提示词长度（均值） | 4285.09 chars |
| 提示词长度（最小/最大） | 663 / 32506 chars |

**各子集统计：**

| 子集 | 样本数 | 提示词均值 | 提示词最小值 | 提示词最大值 |
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
      "id": "056bcccb",
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

*注意：部分内容为显示目的已截断。*

## 提示模板

**提示模板:**
```text
{question}

Think step by step, and when you provide the final answer, please use the prefix "The answer is:" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: "The answer is: 42" or "The answer is: yes". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, "The answer is: (a)".
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `mini` | `bool` | `False` | 评估 BBEH Mini，即官方包含 460 个样本的子集（每项任务 20 个样本）。 |

## 使用方法

### 使用 CLI

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
