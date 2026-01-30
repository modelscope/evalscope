# τ-bench

## 概述

τ-bench（Tau Bench）是一个用于评估通过领域特定 API 工具和政策指南与用户交互的对话式 AI 代理的基准测试。它模拟动态的多轮对话，其中语言模型同时扮演用户和代理角色。

## 任务描述

- **任务类型**：对话代理评估
- **输入**：具有特定目标和约束的用户场景
- **输出**：代理通过 API 工具调用执行操作以完成任务
- **领域**：航空客户服务、零售客户服务

## 主要特性

- 使用 LLM 模拟用户的动态对话仿真
- 领域特定的 API 工具和政策指南
- 真实的客户服务场景
- 测试多轮对话能力
- 评估工具使用和政策合规性

## 评估说明

- **需要安装**：`pip install git+https://github.com/sierra-research/tau-bench`
- **用户模型配置**：需设置用户模拟模型
- 主要指标：基于任务完成奖励的 **准确率（Accuracy）**
- 支持 **航空（airline）** 和 **零售（retail）** 领域
- 使用 **pass@k** 聚合方法进行鲁棒性评估
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau_bench.html)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tau_bench` |
| **数据集 ID** | [tau-bench](https://github.com/sierra-research/tau-bench) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling`, `Reasoning` |
| **指标** | N/A |
| **默认示例数量** | 0-shot |
| **评估分割** | `test` |
| **聚合方式** | `mean_and_pass_hat_k` |

## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

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
    --datasets tau_bench \
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
    datasets=['tau_bench'],
    dataset_args={
        'tau_bench': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```