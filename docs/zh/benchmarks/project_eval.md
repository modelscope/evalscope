# Project Eval

## 概述

评估外部生成的业务 Agent 评测集，使用 rubric 判定和只读终态检查。

## 任务描述

- **任务类型**：业务 Agent 对话，支持可选的 user 多轮脚本。
- **输入**：包含必填 question、可选 user messages、参考答案、rubric 和只读检查的 JSONL 任务。
- **输出**：Agent 回复、trace、rubric 判定和终态检查证据。
- **领域**：拥有本地或远程业务状态的企业业务 Agent。

## 主要特性

- 消费外部编写的评测集；EvalScope 不读取项目源码。
- 支持 SQLite、HTTP/JSON、JSON 文件和文件检查。
- 支持 L0 和 L2 inline trace；没有 trace 时 process 判据不可评估。

## 评测说明

- 指标：task_pass、rubric_score、evidence_coverage；task_pass 为主指标。
- 检查只读；带 must-pass outcome criterion 的任务，任意检查失败即失败。
- 有检查时必须配置 `project_root`；远程状态 API 需要显式 `check_url_prefixes`。
- `trace.mode=inline` 要求内联 steps；全 process rubric 且无 trace 的样本不计入指标并标记 `needs_review`。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准名称** | `project_eval` |
| **数据集 ID** | `project_eval` |
| **论文** | N/A |
| **标签** | `Agent` |
| **指标** | `task_pass`、`rubric_score`、`evidence_coverage` |
| **默认 Shots** | 0-shot |
| **评测 Split** | `N/A` |

## 数据统计

*暂无统计数据。*

## 样本示例

*暂无样本示例。*

## 提示模板

*未定义提示模板。*

## 使用方式

### CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets project_eval \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
    --limit 10  # 正式评测时删除这一行
```

### Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['project_eval'],
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    limit=10,  # 正式评测时删除这一行
)

run_task(task_cfg=task_cfg)
```
