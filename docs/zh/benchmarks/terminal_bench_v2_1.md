# Terminal-Bench-2.1


## 概述

Terminal-Bench v2.1 是 Terminal-Bench 2.0 的改进迭代版本，包含 26 项任务修复，旨在解决错误、调整超时设置以及防止奖励黑客（reward hacking）行为。对于新的评估任务，推荐使用 v2.1 而非 v2.0。

## 任务描述

- **任务类型**：命令行智能体评估
- **输入**：终端任务规范
- **输出**：通过智能体操作完成任务
- **领域**：系统管理、编译、调试、文件操作

## 主要特点

- Terminal-Bench 2.0 的验证迭代版本，包含 26 项任务修复
- 增强了对抗奖励黑客行为的鲁棒性
- 隔离的容器执行环境
- 二进制评分（0/1），支持自动验证
- 支持多种智能体类型（terminus-2, claude-code, codex 等）

## 评估说明

- 需要 **Python>=3.12** 和 `pip install evalscope[terminal_bench]`
- 环境选项：docker, daytona, e2b, modal
- 通过 Harbor 支持 E2B：安装 `harbor[e2b]` 并设置 `extra_params.environment_type='e2b'`，即可在无需本地 Docker 或下载本地任务镜像的情况下运行。请参阅 [沙箱指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html#e2b)。
- 可配置的智能体类型和超时设置
- 最大轮数可配置（默认值：200）
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/terminal_bench.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `terminal_bench_v2_1` |
| **数据集ID** | [latest](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 89 |

## 样例示例

*暂无可用样例示例。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `environment_type` | `str` | `docker` | 运行基准测试的环境类型。可选值：['docker', 'daytona', 'e2b', 'modal'] |
| `agent_name` | `str` | `terminus-2` | 在 Harbor 中使用的智能体类型。仅 terminus-2 使用 evalscope 模型进行推理；其他智能体（claude-code, codex 等）作为独立的 CLI 工具运行，并使用其自身的 API 密钥。可选值：['oracle', 'terminus-2', 'claude-code', 'codex', 'qwen-coder', 'openhands', 'opencode', 'mini-swe-agent'] |
| `timeout_multiplier` | `float` | `1.0` | 超时乘数。如果发生超时错误，请考虑增加此值。 |
| `agent_timeout_sec` | `float` | `None` | 最终智能体超时时间（秒）。不能与 agent_timeout_multiplier 同时使用。 |
| `verifier_timeout_sec` | `float` | `None` | 最终验证器超时时间（秒）。不能与 verifier_timeout_multiplier 同时使用。 |
| `agent_timeout_multiplier` | `float` | `None` | 智能体超时乘数。覆盖 timeout_multiplier 在智能体阶段的设置。 |
| `verifier_timeout_multiplier` | `float` | `None` | 验证器超时乘数。覆盖 timeout_multiplier 在验证器阶段的设置。 |
| `max_turns` | `int` | `200` | 智能体完成任务的最大轮数。 |
| `environment_kwargs` | `dict` | `{}` | 传递给 Harbor EnvironmentConfig 的额外 kwargs。支持的键包括：override_cpus, override_memory_mb, override_storage_mb, override_gpus, force_build, delete, env 等。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets terminal_bench_v2_1 \
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
    datasets=['terminal_bench_v2_1'],
    dataset_args={
        'terminal_bench_v2_1': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
