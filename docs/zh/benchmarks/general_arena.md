# GeneralArena


## 概述

GeneralArena 是一个自定义基准测试，旨在评估大型语言模型在竞争环境中的性能。在该环境中，模型在自定义任务中相互对抗，以确定其相对优势和劣势。

## 任务描述

- **任务类型**：模型对模型竞技场评估
- **输入**：包含多个模型响应的用户提示词
- **输出**：成对比较判断和 ELO 评分
- **重点**：对比式模型评估

## 主要特点

- 使用 LLM-as-judge 进行成对比较
- 计算 ELO 评分以进行模型排名
- 支持自定义模型对进行比较
- 胜率及排名指标
- 可配置基线模型

## 评估说明

- 默认配置使用 **0-shot** 评估
- 需要来自先前评估的模型输出
- 使用 LLM-as-judge 进行质量评估
- 聚合方式：ELO 评分计算
- 详情参见 [Arena User Guide](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `general_arena` |
| **数据集ID** | `general_arena` |
| **论文** | N/A |
| **标签** | `Arena`, `Custom` |
| **指标** | `win_rate` |
| **默认 Shots** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `elo` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

**系统提示：**
```text
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. Include that answer and your comparison in the JSON reasoning field.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

Put your explanation in the JSON reasoning field and your final verdict in the JSON verdict field as one of: A>>B (Assistant A is significantly better), A>B (Assistant A is slightly better), A=B (tie), B>A (Assistant B is slightly better), or B>>A (Assistant B is significantly better).
```

**提示模板：**
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
| `models` | `list[dict]` | `[{'name': 'qwen-plus', 'report_path': 'outputs/20250627_172550/reports/qwen-plus'}, {'name': 'qwen2.5-7b', 'report_path': 'outputs/20250627_172817/reports/qwen2.5-7b-instruct'}]` | 用于竞技场比较的模型条目列表，包含名称和 report_path。 |
| `baseline` | `str` | `qwen2.5-7b` | 用于 ELO 和胜率比较的基线模型名称。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets general_arena \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
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
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    dataset_args={
        'general_arena': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
