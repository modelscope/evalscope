# τ³-bench


## 概述

τ³-bench（Tau Cubed Bench）是 tau-bench 系列的 v1.0.0 版本。它在 τ²-bench 的基础上新增了知识检索领域、原生语音/音频评估，并对现有领域的 75+ 项任务进行了修复。

## 任务描述

- **任务类型**：支持可选知识检索的对话智能体评估
- **输入**：包含复杂目标和多步骤需求的用户场景
- **输出**：遵循策略指南的智能体 API 工具调用动作
- **领域**：航空、零售、电信、银行知识

## 核心特性

- 新增 `banking_knowledge` 领域，包含 97 项任务和 698 份政策/流程文档（RAG）
- 对航空 / 零售 / 银行领域的 75+ 项任务进行了质量修复
- 可插拔检索管道：BM25、稠密嵌入（OpenAI / Qwen）、grep、沙箱 shell、重排序器
- LLM 模拟用户交互，支持多轮对话与工具调用

## 评估说明

- **Python 版本要求**：3.12–3.13
- **安装命令**：`pip install 'tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench@v1.0.0'`
- **不可与 `tau2_bench` 共存**于同一环境（PyPI 包名同为 `tau2`，但版本不同）。二者择一。
- **用户模型配置**：需设置用户模拟模型
- **检索配置（仅限 banking_knowledge）**：默认使用 `bm25`（离线）。可通过 `extra_params.retrieval_config` 切换。其他配置可能需要额外依赖：
  - `bm25` → 包含在 `[knowledge]` 额外依赖中（无需 API 密钥）
  - `openai_embeddings*` → 需设置 `OPENAI_API_KEY`
  - `qwen_embeddings*` → 需设置 `OPENROUTER_API_KEY`
  - `*_reranker` → 同样需要 `OPENAI_API_KEY`
  - `terminal_use` / `alltools*` → 需要 Anthropic `sandbox-runtime`（npm）及 ripgrep / bwrap / socat（详见 tau2 README）
- 主要指标：基于任务完成奖励的 **准确率（Accuracy）**
- 使用 **pass@k** 聚合方式进行鲁棒性评估
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau3_bench.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tau3_bench` |
| **数据集ID** | [evalscope/tau3-bench-data](https://modelscope.cn/datasets/evalscope/tau3-bench-data/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling`, `Reasoning` |
| **指标** | N/A |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `mean_and_pass_hat_k` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 375 |
| 提示词长度（平均） | 39.22 字符 |
| 提示词长度（最小/最大） | 0 / 661 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `airline` | 50 | 135.58 | 29 | 661 |
| `retail` | 114 | 1.95 | 0 | 145 |
| `telecom` | 114 | 55.7 | 51 | 59 |
| `banking_knowledge` | 97 | 14 | 14 | 14 |

## 样例示例

**子集**: `airline`

```json
{
  "input": [
    {
      "id": "333afe68",
      "content": "Testing that agent refuses to proceed with a cancellation that is not allowed even if User mentions that she had been told she didn't need insurance."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "airline",
  "metadata": {
    "id": "0",
    "description": {
      "purpose": "Testing that agent refuses to proceed with a cancellation that is not allowed even if User mentions that she had been told she didn't need insurance.",
      "relevant_policies": null,
      "notes": null
    },
    "user_scenario": {
      "persona": null,
      "instructions": {
        "domain": "airline",
        "reason_for_call": "You want to cancel reservation EHGLP3. \n\nIt may be more than 24 hours after booking, but it is ok because you were out of town for that time.",
        "known_info": "You are Emma Kim.\nYour user id is emma_kim_9957.",
        "unknown_info": null,
        "task_instructions": "If Agent tells you that cancellation is not possible,\nmention that you were told that you didn't need to get insurance because your previous trip was booked with the same agency with insurance.\n\nYou don't want to cancel if you don't get a refund."
      }
    },
    "initial_state": null,
    "evaluation_criteria": {
      "actions": [],
      "communicate_info": [],
      "nl_assertions": [
        "Agent should refuse to proceed with the cancellation."
      ],
      "reward_basis": [
        "DB",
        "COMMUNICATE"
      ]
    },
    "_domain": "airline"
  }
}
```

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `user_model` | `str` | `qwen-plus` | 用于在环境中模拟用户的模型。 |
| `api_key` | `str` | `EMPTY` | 用户模型后端的 API 密钥。 |
| `api_base` | `str` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 用户模型 API 请求的基础 URL。 |
| `generation_config` | `dict` | `{'temperature': 0.0}` | 用户模型模拟的默认生成配置。 |
| `retrieval_config` | `str` | `bm25` | `banking_knowledge` 领域的检索配置名称。常用值包括：no_knowledge, full_kb, golden_retrieval, bm25, openai_embeddings, qwen_embeddings, *_reranker, *_grep, terminal_use, alltools。非知识领域将忽略此参数。 |
| `retrieval_config_kwargs` | `dict` | `{}` | 可选参数，将传递给检索管道。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tau3_bench \
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
    datasets=['tau3_bench'],
    dataset_args={
        'tau3_bench': {
            # subset_list: ['airline', 'retail', 'telecom']  # 可选，评估指定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```