# GeneralArena

## 概述

GeneralArena 是一个自定义基准测试，旨在通过竞争性场景评估大语言模型的性能。在该基准中，多个模型将在自定义任务中相互对战，以确定各自的相对优势与劣势。

## 任务描述

- **任务类型**：模型对战竞技场评估（Model vs Model Arena Evaluation）
- **输入**：包含多个模型回复的用户提示
- **输出**：成对比较判断结果和 ELO 评分
- **重点**：模型间的对比评估

## 主要特性

- 使用 LLM 作为裁判进行成对比较
- 通过 ELO 评分计算模型排名
- 支持自定义模型对进行比较
- 提供胜率（winrate）和排名指标
- 可配置基线模型

## 评估说明

- 默认配置使用 **0-shot** 评估方式
- 需要预先生成的模型输出结果
- 使用 LLM 作为裁判进行质量评估
- 聚合方式：ELO 评分计算
- 详情请参阅 [Arena 用户指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `general_arena` |
| **数据集ID** | `general_arena` |
| **论文** | N/A |
| **标签** | `Arena`, `Custom` |
| **指标** | `winrate` |
| **默认示例数量** | 0-shot |
| **评估分割** | `test` |
| **聚合方式** | `elo` |

## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

**系统提示（System Prompt）：**
```text
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".
```

**提示模板（Prompt Template）：**
```text
<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_1}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_2}
<|The End of Assistant B's Answer|>
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `models` | `list[dict]` | `[{'name': 'qwen-plus', 'report_path': 'outputs/20250627_172550/reports/qwen-plus'}, {'name': 'qwen2.5-7b', 'report_path': 'outputs/20250627_172817/reports/qwen2.5-7b-instruct'}]` | 用于竞技场比较的模型列表，每个条目包含模型名称和报告路径。 |
| `baseline` | `str` | `qwen2.5-7b` | 用于 ELO 和胜率比较的基线模型名称。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets general_arena \
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
    datasets=['general_arena'],
    dataset_args={
        'general_arena': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```