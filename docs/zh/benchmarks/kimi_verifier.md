# Kimi-Vendor-Verifier (Param Compliance)


## 概述

Kimi-Vendor-Verifier 是 Kimi K2 / K2-Thinking 部署的预检合规性检查工具。它会发送合成探测请求，验证供应商 API 是否正确**拒绝**不可变解码参数（``temperature``、``top_p``、``presence_penalty``、``frequency_penalty``、``n``）的非默认值，并**接受**其默认值。若供应商静默接受错误值，则可能导致模型输出质量下降，无法匹配官方 Moonshot AI 的行为。该工具改编自 [Kimi-Vendor-Verifier/verify_params.py](https://github.com/MoonshotAI/Kimi-Vendor-Verifier/blob/main/verify_params.py)。

## 任务描述

- **任务类型**：API 参数合规性探测（部署健康检查）
- **输入**：一条最小化聊天消息，加上一个测试参数和 thinking-mode 的 ``extra_body``
- **输出**：供应商是否接受（HTTP 200）或拒绝（HTTP 400）该请求
- **数据集**：完全合成 —— 无需下载外部数据集；探测请求根据 K2 规范在代码中生成

## 核心特性

- 合成探测集：每个（子集 × thinking）组合包含一个 ``no_param`` 健康探测 + 5 个默认值（应接受）探测 + 5 个错误值（应拒绝）探测
- 三个子集覆盖所有常见的 Kimi 部署形态：
    - ``kimi`` —— 官方 Moonshot SaaS API（``extra_body = {"thinking": {"type": ...}}``）；开启/关闭 thinking
    - ``opensource`` —— vLLM / SGLang / KTransformers 的 chat-template 钩子（``extra_body = {"chat_template_kwargs": {"thinking": ...}}``）；开启/关闭 thinking
    - ``none`` —— 非混合模型；不发送 thinking 参数
- 当预期应拒绝时，HTTP 400 响应被视为成功信号
- 每个探测仅发送一个小型请求；总开销相比完整基准测试可忽略不计

## 评估说明

- 默认配置使用 **0-shot** 合成探测
- 指标：**param_immutable_reject_rate**、**param_default_accept_rate**、**inference_error_rate**
- 仅 HTTP 400（``BadRequestError``）被视为真正的参数拒绝；传输错误（5xx / 超时 / 429）不计入拒绝/接受的分母，而是通过 ``inference_error_rate`` 单独体现，避免不稳定的供应商获得“免费通行证”
- 正确部署的 Kimi K2 供应商应报告两个比率指标均为 **1.0**，且 ``inference_error_rate = 0``；任何低于此标准的结果均表明存在参数强制执行漏洞或传输不稳定
- 对于非 Kimi 模型，预期 ``param_immutable_reject_rate = 0``（无 K2 规范可强制执行），而 ``param_default_accept_rate = 1.0``（合理默认值被接受）
- 可通过 ``dataset_args={'kimi_verifier': {'subset_list': ['kimi']}}``（或 ``opensource`` / ``none``）选择子集

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `kimi_verifier` |
| **数据集ID** | `kimi_verifier` |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling` |
| **指标** | `param_immutable_reject_rate`, `param_default_accept_rate`, `inference_error_rate` |
| **默认Shots** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 55 |
| 提示词长度（平均） | 26 字符 |
| 提示词长度（最小/最大） | 26 / 26 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `kimi` | 22 | 26 | 26 | 26 |
| `opensource` | 22 | 26 | 26 | 26 |
| `none` | 11 | 26 | 26 | 26 |

## 样例示例

**子集**: `kimi`

```json
{
  "input": [
    {
      "id": "03c069db",
      "content": "Say 'OK' and nothing else."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "kimi",
  "metadata": {
    "think_mode": "kimi",
    "thinking": false,
    "param_name": null,
    "test_value": null,
    "expected_reject": false
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
    --datasets kimi_verifier \
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
    datasets=['kimi_verifier'],
    dataset_args={
        'kimi_verifier': {
            # subset_list: ['kimi', 'opensource', 'none']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```