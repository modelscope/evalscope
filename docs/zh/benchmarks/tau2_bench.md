# τ²-bench


## 概述

τ²-bench（Tau Squared Bench）是对原始 τ-bench 的扩展和增强，用于在特定领域场景中评估对话式 AI 智能体，并新增了对电信领域的支持。

## 任务描述

- **任务类型**：高级对话智能体评估  
- **输入**：包含复杂目标和多步骤需求的用户场景  
- **输出**：智能体根据策略指南通过 API 工具调用执行动作  
- **领域**：航空、零售、电信  

## 主要特性

- 扩展领域覆盖范围（在航空和零售基础上新增电信领域）  
- 提升任务复杂度与评估标准  
- 基于大语言模型（LLM）模拟用户交互  
- 支持多轮对话与工具调用  
- 验证策略合规性  

## 评估说明

- **安装要求**：`pip install git+https://github.com/sierra-research/tau2-bench@v0.2.0`  
- **用户模型配置**：需设置用户模拟模型  
- 主要指标：基于任务完成奖励的 **准确率（Accuracy）**  
- 支持 **航空（airline）**、**零售（retail）** 和 **电信（telecom）** 领域  
- 使用 **pass@k** 聚合方法进行鲁棒性评估  
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau2_bench.html)  
- 更新版本 **τ³-bench**（v1.0.0）已作为 `tau3_bench` 基准发布，新增 `banking_knowledge` 领域并修复 75+ 项任务。注意：两个版本不能在同一环境中同时安装。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tau2_bench` |
| **数据集ID** | [evalscope/tau2-bench-data](https://modelscope.cn/datasets/evalscope/tau2-bench-data/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling`, `Reasoning` |
| **指标** | N/A |
| **默认示例数（Shots）** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `mean_and_pass_hat_k` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 278 |
| 提示词长度（平均） | 46.65 字符 |
| 提示词长度（最小/最大） | 0 / 658 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `airline` | 50 | 127.94 | 29 | 658 |
| `retail` | 114 | 1.95 | 0 | 145 |
| `telecom` | 114 | 55.7 | 51 | 59 |

## 样例示例

**子集**：`airline`

```json
{
  "input": [
    {
      "id": "27580cd1",
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
      ]
    }
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

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tau2_bench \
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
    datasets=['tau2_bench'],
    dataset_args={
        'tau2_bench': {
            # subset_list: ['airline', 'retail', 'telecom']  # 可选，用于评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```