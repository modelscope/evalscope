# BBH


## 概述

BBH（BIG-Bench Hard）是从 BIG-Bench 基准测试中精选出的 23 项具有挑战性的任务子集，这些任务之所以被选中，是因为语言模型在最初尝试时表现不佳。这些任务需要复杂的推理能力，并且能从思维链（Chain-of-Thought, CoT）提示中显著受益。

## 任务描述

- **任务类型**：混合型（多项选择题和自由形式）
- **输入**：需要推理的任务特定问题
- **输出**：按指定格式给出的答案
- **子集**：27 项推理任务，分为多项选择题（17 项）和自由形式（10 项）

## 主要特点

- 来自 BIG-Bench 的 27 项具有挑战性的推理任务
- 多项选择题任务：时间序列、歧义消解、逻辑推理等
- 自由形式任务：算术、导航、布尔表达式等
- 每项任务均配有精心设计的思维链（CoT）示例
- 旨在评估高级推理能力

## 评估说明

- 默认配置使用 **3-shot** 并配合 CoT 提示（推荐）
- 每个子集的 CoT 提示已预定义在 `cot_prompts/` 目录中
- 答案应遵循格式："So the answer is [ANSWER]"
- 设置 `few_shot_num=0` 可禁用少样本示例
- 多项选择题答案将被标准化为单个字母（A、B、C 等）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bbh` |
| **数据集ID** | [evalscope/bbh](https://modelscope.cn/datasets/evalscope/bbh/summary) |
| **论文** | N/A |
| **标签** | `Reasoning` |
| **指标** | `acc` |
| **默认少样本数量** | 3-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 6,511 |
| 提示词长度（平均） | 3307.29 字符 |
| 提示词长度（最小/最大） | 1060 / 7885 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `temporal_sequences` | 250 | 3746.18 | 3646 | 3876 |
| `disambiguation_qa` | 250 | 4047.48 | 3993 | 4099 |
| `date_understanding` | 250 | 1550.66 | 1491 | 1641 |
| `tracking_shuffled_objects_three_objects` | 250 | 3257.42 | 3195 | 3316 |
| `penguins_in_a_table` | 146 | 3030.88 | 2922 | 3201 |
| `geometric_shapes` | 250 | 5270.24 | 5201 | 5384 |
| `snarks` | 178 | 3493.68 | 3339 | 3693 |
| `ruin_names` | 250 | 3832.01 | 3781 | 3948 |
| `tracking_shuffled_objects_seven_objects` | 250 | 3598.1 | 3506 | 3682 |
| `tracking_shuffled_objects_five_objects` | 250 | 3419.36 | 3338 | 3489 |
| `logical_deduction_three_objects` | 250 | 3093.32 | 3014 | 3165 |
| `hyperbaton` | 250 | 3433.3 | 3386 | 3486 |
| `logical_deduction_five_objects` | 250 | 3264.38 | 3118 | 3379 |
| `logical_deduction_seven_objects` | 250 | 3434.09 | 3217 | 3633 |
| `movie_recommendation` | 250 | 2489.85 | 2436 | 2613 |
| `salient_translation_error_detection` | 250 | 7401.64 | 7223 | 7885 |
| `reasoning_about_colored_objects` | 250 | 2818.32 | 2572 | 3102 |
| `multistep_arithmetic_two` | 250 | 2596.98 | 2594 | 2600 |
| `navigate` | 250 | 2508.7 | 2452 | 2626 |
| `dyck_languages` | 250 | 2723.8 | 2680 | 2874 |
| `word_sorting` | 250 | 2481.34 | 2397 | 2569 |
| `sports_understanding` | 250 | 1077.42 | 1060 | 1122 |
| `boolean_expressions` | 250 | 1991.7 | 1980 | 1998 |
| `object_counting` | 250 | 1706.66 | 1647 | 1787 |
| `formal_fallacies` | 250 | 5185.5 | 4918 | 5514 |
| `causal_judgement` | 187 | 4877.42 | 4194 | 6311 |
| `web_of_lies` | 250 | 3300.84 | 3267 | 3340 |

## 样例示例

**子集**: `temporal_sequences`

```json
{
  "input": [
    {
      "id": "7d1767c8",
      "content": "Task description: Answer questions about which times certain events could have occurred.\n\nQ: Today, Emily went to the museum. Between what times could they have gone?\nWe know that:\nEmily woke up at 1pm.\nElizabeth saw Emily reading at the libr ... [TRUNCATED] ... \nOptions:\n(A) 6pm to 9pm\n(B) 7am to 11am\n(C) 1pm to 2pm\n(D) 2pm to 6pm\nA: Let's think step by step. Put your final answer in the format of \"So the answer is [ANSWER]\" (without quotes and markdown) where [ANSWER] is the answer to the problem.\n"
    }
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "subset_key": "temporal_sequences",
  "metadata": {
    "task_type": "multiple_choice"
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
Q: {question}
A: Let's think step by step. Put your final answer in the format of "So the answer is [ANSWER]" (without quotes and markdown) where [ANSWER] is the answer to the problem.

```

<details>
<summary>少样本模板</summary>

```text
{fewshot}

Q: {question}
A: Let's think step by step. Put your final answer in the format of "So the answer is [ANSWER]" (without quotes and markdown) where [ANSWER] is the answer to the problem.

```

</details>

## 使用方法

### 使用命令行接口（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bbh \
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
    datasets=['bbh'],
    dataset_args={
        'bbh': {
            # subset_list: ['temporal_sequences', 'disambiguation_qa', 'date_understanding']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```