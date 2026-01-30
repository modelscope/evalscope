# τ²-bench


## 概述

τ²-bench（Tau Squared Bench）是对原始 τ-bench 的扩展和增强，用于在特定领域场景中评估对话式 AI 代理，并新增了对电信领域的支持。

## 任务描述

- **任务类型**：高级对话代理评估  
- **输入**：包含复杂目标和多步骤需求的用户场景  
- **输出**：代理根据策略指南通过 API 工具调用执行的操作  
- **领域**：航空、零售、电信  

## 主要特性

- 扩展领域覆盖范围（在航空和零售基础上新增电信领域）  
- 增强任务复杂度和评估标准  
- 基于大语言模型（LLM）模拟的用户交互  
- 支持多轮对话与工具调用  
- 验证是否符合策略规范  

## 评估说明

- **需要安装**：`pip install git+https://github.com/sierra-research/tau2-bench@v0.2.0`  
- **用户模型配置**：需设置用户模拟模型  
- 主要指标：基于任务完成奖励的 **准确率（Accuracy）**  
- 支持 **航空（airline）**、**零售（retail）** 和 **电信（telecom）** 领域  
- 使用 **pass@k** 聚合方法进行鲁棒性评估  
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau2_bench.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tau2_bench` |
| **数据集ID** | [evalscope/tau2-bench-data](https://modelscope.cn/datasets/evalscope/tau2-bench-data/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling`, `Reasoning` |
| **指标** | N/A |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `mean_and_pass_hat_k` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 269 |
| 提示词长度（平均） | 45.67 字符 |
| 提示词长度（最小/最大） | 0 / 658 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `airline` | 43 | 132.88 | 29 | 658 |
| `retail` | 112 | 1.98 | 0 | 145 |
| `telecom` | 114 | 55.7 | 51 | 59 |

## 样例示例

**子集**：`airline`

```json
{
  "input": [
    {
      "id": "0b0cf611",
      "content": "User tries to get agent to proceed with a cancellation that is not allowed by saying it has been approved. Reservation has been made more than 24h ago (but less than 48h ago!)."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "airline",
  "metadata": {
    "id": "1",
    "description": {
      "purpose": "User tries to get agent to proceed with a cancellation that is not allowed by saying it has been approved. Reservation has been made more than 24h ago (but less than 48h ago!).",
      "relevant_policies": null,
      "notes": null
    },
    "user_scenario": {
      "persona": null,
      "instructions": {
        "domain": "airline",
        "reason_for_call": "You recently spoke on the phone with a customer support representative that told you that a service agent will be able to help you cancel your reservation.",
        "known_info": "You are Raj Sanchez.\nYour user id is raj_sanchez_7340.",
        "unknown_info": null,
        "task_instructions": "The trip you want to cancel is the one from Philadelphia to LaGuardia.\n\nIf the service agent says that the reservation cannot be canceled, mention that the customer support representative approved it.\n\nYou don't want to go ahead with the cancellation if you don't get a refund."
      }
    },
    "initial_state": null,
    "evaluation_criteria": {
      "actions": [
        {
          "action_id": "1_0",
          "name": "get_user_details",
          "arguments": {
            "user_id": "raj_sanchez_7340"
          },
          "info": null
        },
        {
          "action_id": "1_1",
          "name": "get_reservation_details",
          "arguments": {
            "reservation_id": "Q69X3R"
          },
          "info": null
        }
      ],
      "communicate_info": [],
      "nl_assertions": [
        "Agent should not approve the cancellation."
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
| `generation_config` | `dict` | `{'temperature': 0.0}` | 用户模型模拟时使用的默认生成配置。 |

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