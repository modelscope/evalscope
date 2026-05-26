# MiniMax-Vendor-Verifier


## 概述

MiniMax-Vendor-Verifier 是一个用于验证 MiniMax M2 / M2.5 / M2.7 供应商部署正确性的多验证器基准测试。每条提示行可携带一个可选的 ``check_type`` 标签，用于路由到特定的验证器；此外还始终启用 ``error_only_reasoning`` 检测器，以捕获最常见的部署回归问题。本基准测试改编自 [MiniMax-Provider-Verifier](https://github.com/MiniMax-AI/MiniMax-Provider-Verifier)。

## 任务描述

- **任务类型**：供应商部署正确性检查（多维度）
- **输入**：多轮对话消息（可选工具定义）以及每行的路由标签（``check_type``、``expected_tool_call``）
- **输出**：供应商的聊天补全响应，并根据该行所选的验证器进行评分
- **分发逻辑**：未指定 ``check_type`` 的行默认使用 ``tool_calls`` 验证器；指定了 ``check_type`` 的行仅运行所列的验证器

## 核心特性

- 移植了五个上游验证器作为纯函数：
    - ``tool_calls`` — 对参数进行 JSON Schema 验证，并检查数组命令的合理性，同时基于 ``expected_tool_call`` 生成混淆矩阵
    - ``error_only_reasoning``（始终启用）— 标记包含推理但无内容且无工具调用的响应（一种部署回归问题）
    - ``contains_russian_characters_unicode`` — 语言遵循性检查；当响应中出现西里尔字符时判定为失败
    - ``repeat_n_gram`` — 退化重复检测器（任意 3-gram 出现 4 次或以上即视为失败）
    - ``scenario_check`` — 验证模型是否保留声明的 JSON 属性顺序，用于发现会重新排序 ``parameters.properties`` 的供应商
- 报告中每个验证器的分母：``num=0`` 表示该子集中没有行触发此验证器（并非失败）
- 托管数据集保留了上游的 sample.jsonl 文件，以及 M2.5 / M2.7 的每轮基线追踪数据

## 评估说明

- 默认配置使用 **0-shot** 评估；``default`` 子集包含 102 行
- 评估指标：**tool_calls_match_rate**、**schema_accuracy**、**error_only_reasoning_rate**、**language_following_success_rate**、**repeat_ngram_pass_rate**、**scenario_check_pass_rate**
- 根据上游指导，正确部署的供应商应达到：``tool_calls_match_rate ≈ 0.98``、``schema_accuracy ≥ 0.98``、``error_only_reasoning_rate = 0`` 以及 ``scenario_check_pass_rate = 1.0``
- 使用 ``--limit`` 时，较稀有的 ``check_type`` 行（scenario / repeat / language）可能不会全部被采样；请检查各验证器对应的 ``num`` 列

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `minimax_verifier` |
| **数据集ID** | [evalscope/MiniMaxVendorVerifier](https://modelscope.cn/datasets/evalscope/MiniMaxVendorVerifier/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling` |
| **指标** | `tool_calls_match_rate`, `schema_accuracy`, `error_only_reasoning_rate`, `language_following_success_rate`, `repeat_ngram_pass_rate`, `scenario_check_pass_rate` |
| **默认 Shots** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 102 |
| 提示词长度（平均） | 72251.38 字符 |
| 提示词长度（最小/最大） | 16 / 341252 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "6cc50a79",
      "content": "日本ではどのような時にお年玉を渡しますか？",
      "role": "user"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "tools": [],
  "metadata": {
    "check_type": [
      "contains_russian_characters_unicode"
    ],
    "expected_tool_call": null,
    "tools_raw": []
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets minimax_verifier \
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
    datasets=['minimax_verifier'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```