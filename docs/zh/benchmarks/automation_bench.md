# AutomationBench


## 概述

AutomationBench 在销售、营销、运营、支持、财务和人力资源等领域的实际业务工作流中评估智能体。EvalScope 使用 Zapier 官方 Python 包提供的公开任务、模拟的 SaaS 服务以及基于断言的评分机制进行评测。

## 任务描述

- **任务类型**：涵盖 47 个模拟 SaaS 工具的状态化业务工作流。
- **公开数据集**：共 600 个任务，分别来自 `sales`（销售）、`marketing`（营销）、`operations`（运营）、`support`（支持）、`finance`（财务）和 `hr`（人力资源）六个领域，每个领域各 100 个任务。
- **简单基线**：可选的 `simple` 子集包含 200 个基础性的单步或两步任务。其指标使用 `simple_` 前缀，并与公开基准测试分数分开报告。
- **评分方式**：`partial_credit` 表示满足的评分断言所占比例；`pass_rate` 是官方 `task_completed_correctly` 信号的平均值，仅当所有评分断言全部通过时该值才为 1。

## 评测说明

- 评测前请准备 Python 3.13+ 环境并安装指定依赖：
  `python -m pip install "verifiers==0.1.12.dev2" "automation-bench @ git+https://github.com/zapier/AutomationBench.git@a321764ace3cfbe42289e6a13abef2f0f4f56fad"`。
- 默认情况下，EvalScope 会评测全部六个公开领域（共 600 个任务）。可通过 `subset_list` 参数选择特定领域，或使用 `limit` 参数进行小规模运行。
- 需使用基于 API 的 EvalScope 模型，并指定 `openai_api`、`openai_responses_api` 或 `anthropic_api` 作为评测类型。
- 无需 Docker 运行环境、外部数据集下载或真实的 SaaS 凭据，但仍需提供模型 API 凭据。
- 发布的任务是公开的。Zapier 官方排行榜使用的是独立的私有保留集，因此本地公开分数仅具参考意义，无法与排行榜完全一致。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `automation_bench` |
| **数据集ID** | [AutomationBench](https://github.com/zapier/AutomationBench) |
| **论文** | [Paper](https://arxiv.org/abs/2604.18934) |
| **标签** | `Agent`, `FunctionCalling`, `MultiTurn` |
| **指标** | `pass_rate`, `partial_credit`, `error_rate` |
| **默认示例数** | 0-shot |
| **评测划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 600 |
| 提示词长度（平均） | 983.57 字符 |
| 提示词长度（最小/最大） | 586 / 1785 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `sales` | 100 | 832.89 | 586 | 1245 |
| `marketing` | 100 | 953.02 | 612 | 1275 |
| `operations` | 100 | 1259.54 | 787 | 1785 |
| `support` | 100 | 937.56 | 711 | 1577 |
| `finance` | 100 | 990.51 | 709 | 1259 |
| `hr` | 100 | 927.91 | 689 | 1276 |

## 样例示例

**子集**: `sales`

```json
{
  "input": [
    {
      "id": "97a20b04",
      "content": "You are a workflow automation agent. Execute the requested tasks using the available tools. Do not ask clarifying questions - use the information provided and make reasonable assumptions when needed. You have a budget of ~50 tool-using turns  ... [TRUNCATED 47 chars] ...  searches. When summarizing your work in messages or records, list only items you acted on. Do not name, enumerate, or explain items you skipped, excluded, or rejected — handle exclusions silently in the action, not narratively in the output.",
      "source": "input",
      "role": "system"
    },
    {
      "id": "4d75f740",
      "content": "We just closed the Meridian Corp Platform Deal! Mark it as won and route the win notice to the right team per our routing policy. Be sure to follow the latest routing guidelines. Confirm the account tier from the 'Account Hierarchy' spreadshe ... [TRUNCATED 160 chars] ... support-escalation@example.com, executive-team@example.com, sales-team@example.com, smb-team@example.com, vp-sales@example.com\n\nUse Gmail for all email sends. Include the names of affected entities and the relevant amounts in your message(s).",
      "source": "input",
      "role": "user"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "sales",
  "metadata": {
    "record_key": "sales:0",
    "domain": "sales",
    "task": "sales.multi_hop_lookup",
    "official_example_id": 501
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `toolset` | `str` | `api` | 提供给智能体的官方工具风格。选项：['api', 'zapier', 'limited_zapier'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets automation_bench \
    --limit 10  # 正式评测时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['automation_bench'],
    dataset_args={
        'automation_bench': {
            # subset_list: ['sales', 'marketing', 'operations']  # 可选，用于评测特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评测时请删除此行
)

run_task(task_cfg=task_cfg)
```
