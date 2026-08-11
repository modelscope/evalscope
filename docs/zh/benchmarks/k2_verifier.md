# K2-Vendor-Verifier


## 概述

K2-Vendor-Verifier 用于验证第三方部署的 Kimi-K2 是否忠实地复现了官方 Moonshot AI API 的工具调用行为。它使用官方评估提示集对供应商端点进行重放，并将 `finish_reason` 和工具调用载荷（tool-call payloads）与官方基线进行比较。本基准测试改编自 [MoonshotAI/K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier)。

## 任务描述

- **任务类型**：供应商部署正确性检查（工具调用）
- **输入**：包含可用工具定义的多轮对话消息，与上游 K2VV 提示集完全一致
- **输出**：供应商的聊天补全响应（`finish_reason` 和 `tool_calls`）
- **比较方式**：将供应商的行为与数据集中提供的官方 Moonshot AI 基线进行对比

## 核心特性

- 使用官方的 2,000 行 K2-Thinking 样本集（占上游测试集的 50%）
- 报告 K2VV 主要指标 `trigger_similarity` —— 工具调用决策相对于官方基线的 F1 分数
- 对触发的工具调用参数进行 JSON Schema 合法性校验
- 提供原始计数以供合理性检查（`count_finish_reason_tool_calls`、`count_successful_tool_call`）
- 托管的数据集保留了官方的 `finish_reason` 和 `tool_calls`，便于未来指标进行载荷级别的保真度比较

## 评估说明

- 默认配置采用 **0-shot** 评估；每条样本已包含多轮上下文
- 评估指标：**trigger_similarity**、**schema_accuracy**、**count_finish_reason_tool_calls**、**count_successful_tool_call**
- 根据上游 K2VV README，`trigger_similarity` ≥ 0.73 被视为大致可接受的阈值
- 当前仅发布 `k2_thinking` 子集（K2-0905 将在上游发布后跟进）
- 上游基线中少数历史助手消息的 `tool_calls.arguments` 包含格式错误的 JSON；适配器在加载时会对其进行清理

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `k2_verifier` |
| **数据集ID** | [evalscope/K2VendorVerifier](https://modelscope.cn/datasets/evalscope/K2VendorVerifier/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling` |
| **指标** | `trigger_similarity`, `schema_accuracy`, `count_finish_reason_tool_calls`, `count_successful_tool_call` |
| **默认Shots** | 0-shot |
| **评估分割** | `test` |
| **聚合方式** | `f1` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets k2_verifier \
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
    datasets=['k2_verifier'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
