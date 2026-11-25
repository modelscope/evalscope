# Terminal-Bench-4.0


## 概述

Terminal-Bench 4.0 使用官方 Harbor 框架评估智能体在 66 个具有挑战性的终端任务上的表现。它校准了任务资源，修复了任务指令和验证器，并移除了不再提供有效区分度的任务。

## 任务描述

- **任务类型**：命令行智能体评估
- **输入**：任务指令和一个隔离的运行时环境
- **输出**：智能体生成的产物及任务完成情况，由官方验证器进行检查
- **领域**：软件工程、机器学习、安全、科学和运维

## 主要特点

- 固定上游数据集版本 **4.0.0**，采用内容寻址的任务版本
- 包含 66 个任务，其中 11 个为多容器任务，3 个需要 H100 GPU
- 所有任务均配备独立的验证器环境，包括一个带有 GPU 验证器的任务
- 官方设定每个任务的智能体超时时间为 8 小时；针对特定任务设有 CPU、内存和验证器预算
- 支持可配置的 Harbor 智能体、显式任务名称过滤以及重复评估

## 评估说明

- 需要 **Python>=3.12**、**Harbor>=0.14.0,<1.0.0** 以及 `pip install 'evalscope[terminal_bench]'`
- 分数为有效官方验证器奖励（0/1）的平均值；失败的试验和无效的奖励不计入有效分数
- 默认使用 Docker 和 terminus-2；terminus-2 使用配置的 EvalScope 模型，而外部 CLI 智能体使用其自身的 API 凭证
- 完整评估需要一个支持 H100 GPU 访问和多容器支持的 Docker 主机；Modal 是一个替代方案，需配合 `harbor[modal]` 和账户凭证使用
- 任务资源和超时时间继承自上游；max_turns 默认为 None，使用 Harbor 智能体的默认值
- 使用 task_names 进行显式调试子集选择，使用 repeats 进行重复试验；子集结果和旧版基准测试的结果不能直接与完整的 4.0 评估结果进行比较
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/terminal_bench.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `terminal_bench_v4` |
| **数据集ID** | [4](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/4) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 66 |

## 样例示例

*无可用样例示例。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `environment_type` | `str` | `docker` | 运行基准测试的环境类型。选项：['docker', 'daytona', 'e2b', 'modal'] |
| `agent_name` | `str` | `terminus-2` | 在 Harbor 中使用的智能体类型。只有 terminus-2 使用 evalscope 模型进行推理；其他智能体（如 claude-code, codex 等）作为独立的 CLI 工具运行，并使用其自身的 API 密钥。选项：['oracle', 'terminus-2', 'claude-code', 'codex', 'qwen-coder', 'openhands', 'opencode', 'mini-swe-agent'] |
| `timeout_multiplier` | `float` | `1.0` | 超时倍数。如果发生超时错误，请考虑增加此值。 |
| `agent_timeout_sec` | `float` | `None` | 最终智能体超时时间（秒）。不能与 agent_timeout_multiplier 组合使用。 |
| `verifier_timeout_sec` | `float` | `None` | 最终验证器超时时间（秒）。不能与 verifier_timeout_multiplier 组合使用。 |
| `agent_timeout_multiplier` | `float` | `None` | 智能体超时倍数。覆盖智能体阶段的 timeout_multiplier。 |
| `verifier_timeout_multiplier` | `float` | `None` | 验证器超时倍数。覆盖验证器阶段的 timeout_multiplier。 |
| `max_turns` | `int` | `None` | 最大智能体轮次。None 表示使用 Harbor 智能体的默认值。 |
| `environment_kwargs` | `dict` | `{}` | 传递给 Harbor EnvironmentConfig 的额外 kwargs。支持的键包括：override_cpus, override_memory_mb, override_storage_mb, override_gpus, force_build, delete, env 等。 |
| `task_names` | `list` | `None` | 可选的 Harbor 任务名称或 glob 模式，例如 terminal-bench/ctr-optimization。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets terminal_bench_v4 \
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
    datasets=['terminal_bench_v4'],
    dataset_args={
        'terminal_bench_v4': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
