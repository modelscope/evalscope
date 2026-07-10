# ResearchRubrics


## 概述

ResearchRubrics 用于评估深度研究（Deep Research）智能体在真实、开放式研究任务上的表现。每个任务包含一个用户提示，以及由专家编写的细粒度评分标准（rubrics），涵盖显性和隐性需求、信息综合、参考文献、沟通质量以及指令遵循等方面。

## 任务描述

- **任务类型**：多轮研究智能体 / 长篇报告生成
- **输入**：一个开放式研究提示
- **输出**：通过迭代使用工具生成的 Markdown 格式研究报告
- **数据集**：101 个任务，共包含 2,593 项加权评分标准
- **评估指标**：二元评分标准符合度得分（Binary rubric compliance score）

## 智能体运行时

- 默认使用 EvalScope 内置的 AgentLoop，采用 ``function_calling`` 策略，并默认提供 ``bash`` 工具。
- 默认的 bash 工具通过 ``LocalAgentEnvironment`` 在每个样本的临时目录中运行，并使用主机网络。该环境**不是安全沙箱**：绝对路径仍可访问主机文件系统。请勿在共享或敏感机器上对不可信模型使用默认运行时。
- 可通过 ``dataset_args.extra_params.strategy`` 将策略设为 ``react``。两种内置策略均需原生函数调用支持；ReAct 策略还会额外注入“思考 → 行动 → 观察”（Think → Act → Observation）的系统提示。
- 来自 ``NativeAgentConfig.mcp_servers`` 的可选 MCP 服务器会与 bash 工具合并使用。若使用 ``ExternalAgentConfig``，则提示将通过 EvalScope 的外部智能体桥接器路由。
- 对于本基准测试所拥有的 AgentLoop，策略和最大步数通过 ``dataset_args.extra_params`` 配置；``NativeAgentConfig`` 中对应的字段不会被使用。
- 如果原生循环在达到 ``max_steps`` 后仍在调用工具，基准测试会再进行一次无工具调用的最终模型请求，以确保已收集的研究内容能保存为可审查的 Markdown 报告。

## 评估说明

- ResearchRubrics 要求显式配置 ``judge_model_args``，且 ``judge_strategy`` 必须为 ``'auto'`` 或 ``'llm'``。论文推荐使用 Gemini 2.5 Pro 作为评判模型，但未硬编码任何特定提供商或模型。
- 每项评分标准独立评分：满足（1）或不满足（0），与公开的二元评分器一致。论文中使用的三元评分不可直接比较。
- 负权重标准在出现不良行为时会从分子中减去相应分数，得分不会被截断。
- 当报告长度超过配置的评判模型上下文阈值时，将使用官方的分块-证据-综合（chunk-evidence-synthesis）方法进行评估。
- 完整运行需执行 2,593 次评分标准评估，成本较高。此外，涉及当前事件的任务对评估时的日期和可用网络资源敏感。

## 配置

- ``strategy``：``function_calling``（默认）或 ``react``
- ``max_steps``：默认为 50
- ``judge_context_limit``：150,000 个估算 token
- ``judge_chunk_size``：100,000 个估算 token
- ``judge_retries``：每次评判请求最多重试 3 次

评判模型必须显式配置。例如：

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='YOUR_AGENT_MODEL',
    datasets=['researchrubrics'],
    judge_strategy='llm',
    judge_model_args={
        'model_id': 'YOUR_JUDGE_MODEL',
        'api_url': 'OPENAI_COMPATIBLE_JUDGE_URL',
        'api_key': 'YOUR_JUDGE_API_KEY',
        'generation_config': {'temperature': 0.0},
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
| **数据集 ID** | [evalscope/researchrubrics](https://modelscope.cn/datasets/evalscope/researchrubrics/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2511.07685) |
| **标签** | `Agent`, `MultiTurn`, `Reasoning`, `Retrieval` |
| **指标** | `compliance_score` |
| **默认示例数** | 0-shot |
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
      "id": "aeaec74a",
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

*注：部分内容因显示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `strategy` | `str` | `function_calling` | 内置 AgentLoop 使用的智能体策略。选项：['function_calling', 'react'] |
| `max_steps` | `int` | `50` | 每个样本的最大智能体步数。 |
| `judge_context_limit` | `int` | `150000` | 切换到分块评判前的估算 token 上限。 |
| `judge_chunk_size` | `int` | `100000` | 发送给评判模型的每个文档分块的最大估算 token 数。 |
| `judge_retries` | `int` | `3` | 每次评分标准评判请求及 JSON 解析的最大重试次数。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets researchrubrics \
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
    datasets=['researchrubrics'],
    dataset_args={
        'researchrubrics': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
