# τ²-bench

## 简介

τ²-bench（Tau Squared Bench）是 τ-bench 的扩展与增强版本，包含一系列代码修复，并新增了电信（telecom）领域的故障排查场景。它面向评估大语言模型在动态对话代理中的工具使用与策略遵循能力。

- 项目地址：https://github.com/sierra-research/tau2-bench
- 重要说明：τ²-bench 是最新版本，建议优先使用 τ²-bench 进行评测。

核心特点：
- 动态交互：模拟真实用户与 AI 代理的多轮对话
- 工具集成：代理需要合理使用提供的 API 工具
- 策略遵循：代理需要遵循业务策略和指导原则
- 领域扩展：在航空、零售基础上新增电信故障排查场景
- 可靠性提升：在 τ-bench 基础上进行了代码修复与稳定性改进

支持的评测领域：
- airline：航空公司客服
- retail：零售客服
- telecom：电信客服（新增，包含网络/套餐/账单/故障排查等）

## 安装依赖

```bash
pip install evalscope
# 安装 tau2-bench
pip install "git+https://github.com/sierra-research/tau2-bench@v0.2.0"
```

```{important}
- 数据集由 evalscope 自动从 ModelScope 拉取（数据集 ID：evalscope/tau2-bench-data），并自动设置 TAU2_DATA_DIR。
- 仅支持通过 API 服务评测被测模型（建议用 vLLM 等框架将本地模型先以服务形式暴露）。
```

## 使用方法

以 qwen-plus 为例。官方榜单通常使用 `user model = gpt-4.1-2025-04-14`，如需对齐榜单请将 user_model 配置为 gpt-4.1-2025-04-14 并提供对应 API Key 与 Base URL。

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    # 被测代理模型（Agent Model）
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # 使用 OpenAI 兼容的服务评测

    datasets=['tau2_bench'],
    dataset_args={
        'tau2_bench': {
            'subset_list': ['airline', 'retail', 'telecom'],  # 支持三个领域
            'extra_params': {
                # 用户模拟模型（User Model），用于驱动对话环境
                'user_model': 'qwen-plus',  # 如需对齐官方榜单可改为 'gpt-4.1-2025-04-14'
                'api_key': os.getenv('DASHSCOPE_API_KEY'),
                'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'generation_config': {
                    'temperature': 0.7,
                }
            }
        }
    },

    eval_batch_size=5,  # 评测并发大小
    limit=5,  # 快速试跑建议保留，正式评测可去掉
    generation_config={
        'temperature': 0.6,
    },
)

run_task(task_cfg)
```

提示：
- 若使用 gpt-4.1-2025-04-14 作为用户模拟模型，请设置：
  - extra_params.user_model='gpt-4.1-2025-04-14'
  - extra_params.api_base='https://api.openai.com/v1'
  - extra_params.api_key=<OPENAI_API_KEY>

## 评测流程

1) 任务初始化：为代理提供领域 API 工具与策略指导  
2) 用户模拟：用户模型按场景产生自然请求  
3) 代理响应：被测模型按工具与策略生成响应  
4) 多轮交互：持续对话直至任务完成或失败  
5) 结果评估：基于任务完成度与策略遵循度计分

## 评估维度

- 是否完成用户目标（任务完成度）
- 是否正确调用必要的 API 工具
- 是否遵循业务策略与约束

## 领域特性

- 航空（Airline）
  - 工具：航班查询、改签、座位、退改等
  - 典型任务：改签、座位升级、行李问题处理
- 零售（Retail）
  - 工具：商品/订单/库存/支付等
  - 典型任务：商品推荐、订单追踪、退换货处理
- 电信（Telecom，新增）
  - 工具：网络诊断、套餐变更、停复机、故障工单等
  - 典型任务：网络连接问题、账单争议、套餐升级与排障

## 结果示例

```text
+-----------+------------+-------------+----------+-------+---------+---------+
| Model     | Dataset    | Metric      | Subset   |   Num |   Score | Cat.0   |
+===========+============+=============+==========+=======+=========+=========+
| qwen-plus | tau2_bench | mean_Pass^1 | airline  |    10 |     0.6 | default |
+-----------+------------+-------------+----------+-------+---------+---------+
| qwen-plus | tau2_bench | mean_Pass^1 | retail   |    10 |     0.7 | default |
+-----------+------------+-------------+----------+-------+---------+---------+
| qwen-plus | tau2_bench | mean_Pass^1 | telecom  |    10 |     0.8 | default |
+-----------+------------+-------------+----------+-------+---------+---------+
| qwen-plus | tau2_bench | mean_Pass^1 | OVERALL  |    30 |     0.7 | -       |
+-----------+------------+-------------+----------+-------+---------+---------+ 
```

## 指标说明

- Pass^1：首次尝试即完成任务的比例（越高越好）
  - 体现单轮对话内的工具使用正确性、策略遵循与目标达成度
