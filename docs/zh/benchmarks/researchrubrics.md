# ResearchRubrics


## 概述

ResearchRubrics 用于评估深度研究（Deep Research）智能体在真实、开放式研究任务上的表现。每个任务包含一个用户提示，以及由专家编写的细粒度评分标准（rubrics），涵盖显性和隐性需求、信息综合、参考文献、沟通质量以及指令遵循等方面。

## 任务描述

- **任务类型**：多轮研究智能体 / 长篇报告生成
- **输入**：一个开放式研究提示
- **输出**：通过迭代使用工具生成的 Markdown 研究报告
- **数据集**：101 个任务，共包含 2,593 条带权重的评分标准
- **评估指标**：二元评分标准符合度得分（Binary rubric compliance score）

## 核心特性

- 包含 101 个开放式深度研究任务，配以 2,593 条由专家编写并赋予权重的评分标准。
- 评分标准覆盖显性与隐性要求、信息综合、参考文献、沟通质量及指令遵循等多个维度，每条标准独立评分。
- 负权重标准用于捕捉不良行为，当此类行为出现时会从总分中扣除相应分数。
- 当报告长度超过配置的评判器上下文阈值时，将采用官方的分块-证据-综合（chunk-evidence-synthesis）流程进行评分。

## 智能体环境

- 默认使用 EvalScope 内置的智能体环境，无需显式提供 ``agent_config``。智能体可通过 ``bash`` 访问网络、收集信息并生成最终报告。
- 默认环境使用主机网络和临时工作目录，但不提供完整的文件系统隔离。请勿在共享或敏感机器上运行不可信模型。
- 默认策略为 ``function_calling``，最多执行 50 步。可通过 ``NativeAgentConfig`` 覆盖策略或步数限制；也可选用 ``react`` 策略。两种策略均需模型原生支持函数调用。
- 可通过 ``NativeAgentConfig`` 添加专用搜索或网页抓取工具，或使用 ``ExternalAgentConfig`` 在其他智能体框架中运行任务。
- 若达到步数上限，模型将被要求基于已收集的信息生成最终报告，以便后续评审和打分。

## 评估说明

- ResearchRubrics 要求配置 ``judge.models``，且 ``judge.strategy`` 必须为 ``'auto'`` 或 ``'llm'``。论文推荐使用 Gemini 2.5 Pro 作为评判器，但未硬编码任何特定提供商或模型。
- 每条评分标准独立判为“满足”（1）或“不满足”（0），与公开的二元评分器一致。论文中使用的三元评分不可直接比较。
- 当出现不良行为时，负权重标准会从分子中扣除对应分数，且最终得分不会被截断（clipped）。
- 当报告长度超过配置的评判器上下文阈值时，将采用官方的分块-证据-综合方法进行评估。
- 完整运行需执行 2,593 次评分标准评估，成本较高。此外，涉及当前事件的任务对评估时的日期和可用网络资源较为敏感。

## 配置

- ``judge_context_limit``: 150,000 个估算 token
- ``judge_chunk_size``: 100,000 个估算 token

评判器必须显式配置。例如：

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='YOUR_AGENT_MODEL',
    datasets=['researchrubrics'],
    judge={
        'strategy': 'llm',
        'models': {
            'model_id': 'YOUR_JUDGE_MODEL',
            'api_url': 'OPENAI_COMPATIBLE_JUDGE_URL',
            'api_key': 'YOUR_JUDGE_API_KEY',
            'generation_config': {'temperature': 0.0},
        },
    },
    limit=1,
))
```

资源链接：[论文](https://arxiv.org/abs/2511.07685) |
[GitHub](https://github.com/scaleapi/researchrubrics) |
[数据集](https://modelscope.cn/datasets/evalscope/researchrubrics)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `researchrubrics` |
| **数据集ID** | [evalscope/researchrubrics](https://modelscope.cn/datasets/evalscope/researchrubrics/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2511.07685) |
| **标签** | `Agent`, `MultiTurn`, `Reasoning`, `Retrieval` |
| **指标** | `compliance_score` |
| **默认示例数量** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 101 |
| 提示词长度（平均） | 555.35 字符 |
| 提示词长度（最小/最大） | 102 / 1747 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "f1132c85",
      "content": "I want to create a plan for July 4, 2025, i.e., Independence Day in Washington DC. I would like an itinerary of all the things to do and all the activities that are planned for Independence Day. Create a plan for the whole day and also extend it to the weekend, if required. Provide some reviews or explain why one should visit the place or engage in the activity. Add any additional information that is required."
    }
  ],
  "target": "[{\"criterion\": \"The response covers the period from 9:00 AM or earlier through at least 10:00 PM on 4 July 2025\", \"weight\": 5.0, \"axis\": \"Explicit Criteria\"}, {\"criterion\": \"The response contains clear section headers for parts of the schedul ... [TRUNCATED 4470 chars] ...  events from years other than 2025 (e.g., seeing a miltiary parade, information about \\\"A Capital Fourth\\\" for 2024, information the parade route for 2023, referencing a cancellation from 2020).\", \"weight\": -4.0, \"axis\": \"Explicit Criteria\"}]",
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "bash",
      "description": "Execute a bash command inside the sandbox environment. Returns the combined stdout / stderr output of the command.",
      "parameters": {
        "properties": {
          "command": {
            "type": "string",
            "description": "The bash command to execute."
          },
          "timeout": {
            "type": "number",
            "description": "Maximum execution time in seconds (default: 60).",
            "default": 60
          }
        },
        "required": [
          "command"
        ]
      }
    }
  ],
  "metadata": {
    "sample_id": "6847465956a0f6376a605427",
    "domain": "Current Events",
    "conceptual_breadth": "Simple",
    "logical_nesting": "Intermediate",
    "exploration": "Medium"
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `judge_context_limit` | `int` | `150000` | 切换至分块评分前的估算 token 上限。 |
| `judge_chunk_size` | `int` | `100000` | 发送给评判器的每个文档分块的最大估算 token 数。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets researchrubrics \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":50}' \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['researchrubrics'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=50,
    ),
    dataset_args={
        'researchrubrics': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
